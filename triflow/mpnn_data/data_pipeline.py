# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datetime
from multiprocessing import cpu_count
from typing import Mapping, Optional, Sequence, Any

import numpy as np
import torch
import random

import torch.nn as nn


from triflow.np import residue_constants, protein
from triflow.np import multimer_residue_constants as rc

from triflow.utils.all_atom_multimer import atom14_to_atom37 


FeatureDict = Mapping[str, np.ndarray]


def make_pdb_features_multimer(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    pdb_feats["asym_id"] = protein_object.chain_index
    pdb_feats["noiseless_all_atom_positions"] = pdb_feats["all_atom_positions"]
    pdb_feats["residue_index"] = protein_object.residue_index
    if(is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats




def make_sequence_features(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    # features["residue_index"] = np.array(range(num_res), dtype=np.int32)    
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features



def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])


def make_protein_features(
    protein_object: protein.Protein, 
    description: str,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)

    return pdb_feats


def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if(is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats






class DataPipeline:
    """Assembles input features."""




        
################# start of multimer #################

    def process_pdb_multimer(self, 
                            pdb_path: str,
                            is_distillation: bool = False,
                            chain_id: Optional[str] = None,                              
                            ):
        #used during inference
        with open(pdb_path, 'r') as f:
            pdb_string = f.read()
            
        protein_object = protein.from_pdb_string(pdb_string, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features_multimer(
            protein_object, 
            description, 
            is_distillation=is_distillation
        )

        pdb_feats["backbone_rigid_mask"] = pdb_feats["all_atom_mask"][...,1] #pick ca mask        
        pdb_feats["seq_mask"] = np.ones(pdb_feats["backbone_rigid_mask"].shape[0])
        pdb_feats["diffuse_mask"] = np.ones(pdb_feats["backbone_rigid_mask"].shape[0]).astype(np.float32)
        pdb_feats["avg_distances"] = np.nan_to_num(self.compute_avg_distance(pdb_feats['all_atom_positions'][:,1,:][None]))[0].astype(np.float32)  #calculate closest atoms based on Ca distances

        
        
        return pdb_feats


    def loader_pdb(self, item,params):
        #this function is to read proteinmpnn dataset file format
        pdbid,chid = item[0].split('_')
        PREFIX = "%s/pdb/%s/%s"%(params['DIR'],pdbid[1:3],pdbid)
        
        # load metadata
        if not os.path.isfile(PREFIX+".pt"):
            return {'seq': np.zeros(5)}
        meta = torch.load(PREFIX+".pt")
        asmb_ids = meta['asmb_ids']
        asmb_chains = meta['asmb_chains']
        chids = np.array(meta['chains'])
        
        # find candidate assemblies which contain chid chain
        asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                            if chid in b.split(',')])

        # if the chains is missing is missing from all the assemblies
        # then return this chain alone
        if len(asmb_candidates)<1:
            chain = torch.load("%s_%s.pt"%(PREFIX,chid))
            L = len(chain['seq'])
            return {'seq'    : chain['seq'],
                    'xyz'    : chain['xyz'],
                    'idx'    : torch.zeros(L).int(),
                    'masked' : torch.Tensor([0]).int(),
                    'label'  : item[0]}

        # randomly pick one assembly from candidates
        
        asmb_i = random.sample(list(asmb_candidates), 1)
        # asmb_i = [(list(asmb_candidates)[0])] #temporary for mpnn validation


        # indices of selected transforms
        idx = np.where(np.array(asmb_ids)==asmb_i)[0]

        # load relevant chains
        chains = {c:torch.load("%s_%s.pt"%(PREFIX,c))
                for i in idx for c in asmb_chains[i].split(',')
                if c in meta['chains']}

        # generate assembly
        asmb = {}
        for k in idx:

            # pick k-th xform
            xform = meta['asmb_xform%d'%k]
            u = xform[:,:3,:3]
            r = xform[:,:3,3]

            # select chains which k-th xform should be applied to
            s1 = set(meta['chains'])
            s2 = set(asmb_chains[k].split(','))
            chains_k = s1&s2

            # transform selected chains 
            for c in chains_k:
                try:
                    xyz = chains[c]['xyz']
                    xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                    asmb.update({(c,k,i):xyz_i for i,xyz_i in enumerate(xyz_ru)})
                except KeyError:
                    return {'seq': np.zeros(5)}

        # select chains which share considerable similarity to chid
        seqid = meta['tm'][chids==chid][0,:,1]
        homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                    if seqid_j>params['HOMO']])
        # stack all chains in the assembly together
        seq,xyz,idx,masked = "",[],[],[]
        seq_list = []
        for counter,(k,v) in enumerate(asmb.items()):
            seq += chains[k[0]]['seq']
            seq_list.append(chains[k[0]]['seq'])
            xyz.append(v)
            idx.append(torch.full((v.shape[0],),counter))
            if k[0] in homo:
                masked.append(counter)

        return {'seq'    : seq,
                'xyz'    : torch.cat(xyz,dim=0),
                'idx'    : torch.cat(idx,dim=0),
                'masked' : torch.Tensor(masked).int(),
                'label'  : item[0]}

    

    def process_multimer_chains(self, item,params, mode):
        
        features = {}
    
        protein = self.loader_pdb(item[:2], params)

        sequence  = protein["seq"]

        # if len(sequence) > 100000:
        if len(sequence) > 50000:
            sequence = np.array([0, 0, 0, 0, 0])
        
        if isinstance(sequence, np.ndarray):

            # mask and set everything to zero when we encounter an error with loader_pdb
            # willing to waste computation
            dummy_sequence = "AAAAA"
            features["aatype"] = residue_constants.sequence_to_onehot(
                    sequence=dummy_sequence,
                    mapping=residue_constants.restype_order_with_x,
                    map_unknown_to_x=True,
                )
            
            num_res = len(dummy_sequence)
            features["residue_index"] = np.array(range(num_res), dtype=np.int32)
            features["asym_id"] = np.zeros((5, ))
            features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
            features["sequence"] = np.array( [dummy_sequence.encode("utf-8")], dtype=object)

            # features["all_atom_positions"] =  np.zeros((5, 14, 3))
            # features["all_atom_mask"] = np.zeros((5, 14))

            features["all_atom_positions"] =  np.zeros((5, 37, 3))
            features["all_atom_mask"] = np.zeros((5, 37))



            features["noiseless_all_atom_positions"] = features["all_atom_positions"]
            features["backbone_rigid_mask"] = np.zeros((5))

            features["resolution"] = np.array([0.]).astype(np.float32)
            features["is_distillation"] = np.array(0.).astype(np.float32)
            
            features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)

            # features['avg_distances'] = np.zeros((num_res,), dtype = np.float32)
            features['avg_distances'] = np.zeros((num_res,3), dtype = np.float32)
            # features["noise_label"] = np.full((num_res,), 0, dtype=np.float32)
            features["noise_label"] = np.array(0.).astype(np.float32)            


            # features["noise_label"] = nn.functional.one_hot(
            #     torch.zeros(num_res, dtype=torch.long), num_classes=2
            # ).float().numpy()

            features["mask_everything"] = np.array(1.).astype(np.float32) #mask all the positions in a sequence 

        else:

            features["aatype"] = residue_constants.sequence_to_onehot(
                    sequence=sequence,
                    mapping=residue_constants.restype_order_with_x,
                    map_unknown_to_x=True,
                )
            
            num_res = len(sequence)

            features["residue_index"] = np.array(range(num_res), dtype=np.int32)
            features["asym_id"] = np.array(protein["idx"], dtype = np.int32)
            
            features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
            features["sequence"] = np.array( [sequence.encode("utf-8")], dtype=object)
            

            aatype = torch.tensor(features["aatype"], dtype=torch.long) #need to get aatype
            aatype = torch.argmax(aatype, dim=1)            
            atom_pos, atom_mask = atom14_to_atom37(protein['xyz'], aatype)
            features["all_atom_positions"] = atom_pos.numpy().astype(np.float32) #this is atom37 positions
            features["all_atom_mask"] = ((1 - torch.isnan(atom_pos).to(torch.int)[...,-1]) * atom_mask).numpy().astype(np.int32) #need to mask the atoms that dont exist and also based on aatype for atom37
            

            # features["all_atom_positions"] = np.array(protein["xyz"]).astype(np.float32)
            # features["all_atom_mask"] = np.array(1 - torch.isnan(protein['xyz']).to(torch.int)[...,-1])

            backbone_rigid_mask = 1 - torch.isnan(protein['xyz']).to(torch.int)[:,:4, -1] #pick the first three atom14 positions
            backbone_rigid_mask = (backbone_rigid_mask.min(dim=1)[0] != 0).int()

            features["backbone_rigid_mask"] = np.array(backbone_rigid_mask).astype(np.float32)

            #mask seqs where positions do not exist, his tag and X res
            res_mask = np.zeros_like(backbone_rigid_mask).astype(np.float32)

            for idx in list(np.unique(protein['idx'])):
                res = np.argwhere(protein['idx'] == idx)
                initial_sequence= "".join(list(np.array(list(protein['seq']))[res][0,]))
                if initial_sequence[-6:] == "HHHHHH":
                    res = res[:,:-6]                    
                if initial_sequence[0:6] == "HHHHHH":
                    res = res[:,6:]
                if initial_sequence[-7:-1] == "HHHHHH":
                    res = res[:,:-7]
                if initial_sequence[-8:-2] == "HHHHHH":
                    res = res[:,:-8]
                if initial_sequence[-9:-3] == "HHHHHH":
                    res = res[:,:-9]
                if initial_sequence[-10:-4] == "HHHHHH":
                    res = res[:,:-10]
                if initial_sequence[1:7] == "HHHHHH":
                    res = res[:,7:]
                if initial_sequence[2:8] == "HHHHHH":
                    res = res[:,8:]
                if initial_sequence[3:9] == "HHHHHH":
                    res = res[:,9:]
                if initial_sequence[4:10] == "HHHHH":
                    res = res[:,10:]
                    

                res_mask[res[0]] = 1

            res_mask = res_mask * features["backbone_rigid_mask"]  
            unk_pos = np.where(features["aatype"][:, -1] == 1)[0] #positions with unknown residues
            res_mask[unk_pos] = 0
            
            features["seq_mask"] = res_mask.astype(np.int32) 
            features["noiseless_all_atom_positions"] = features["all_atom_positions"]
            features["noise_label"] = np.array(0.).astype(np.float32)

            #lets add some gaussian noise to the atom positions 
            if mode == 'train':
                if torch.rand(1) < 0.5:
                    z = np.random.randn(*features["all_atom_positions"].shape)
                    std = 0.2 #random number between 0 and 0.2
                    features["all_atom_positions"] = features["all_atom_positions"] + (z * std)
                    features["noise_label"] =  np.array(1.).astype(np.float32)
            
            
            features["resolution"] = np.array([0.]).astype(np.float32)
            features["is_distillation"] = np.array(0.).astype(np.float32)

            #random stuff
            features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
            features["mask_everything"] = np.array(0.).astype(np.float32) #do not mask all the positions in a sequence

            #lets also get the decoding order information                 
            features['avg_distances'] = np.nan_to_num(self.compute_avg_distance(features['all_atom_positions'][:,1,:][None]))[0].astype(np.float32)  #calculate closest atoms based on Ca distances            
                        

                    
        return features


    def compute_avg_distance(self, positions):
        """
        Computes the average distance to the 8 closest residues for each residue.

        Args:
            positions (numpy.ndarray): Array of shape (batch_size, n_res, 3) containing the positions.

        Returns:
            numpy.ndarray: Array of shape (batch_size, n_res) containing the average distances.
        """
        batch_size, n_res, _ = positions.shape

        # Compute pairwise squared distances
        diff = positions[:, :, None, :] - positions[:, None, :, :]  # Shape: (batch_size, n_res, n_res, 3)
        squared_distances = np.sum(diff ** 2, axis=-1)  # Shape: (batch_size, n_res, n_res)
        distances = np.sqrt(squared_distances + 1e-8)  # Add epsilon to avoid sqrt(0)

        # Exclude self-distances by setting diagonals to infinity
        mask = np.eye(n_res, dtype=bool)[None, :, :]  # Shape: (1, n_res, n_res)
        distances[mask] = np.inf

        # Find the 8 smallest distances for each residue
        k0 = 4
        k1 = 8
        k2 = 16

        #if n_res less than 16 then compute average of distances of n_res
        if n_res <= 16:
            k2 = n_res - 1

        topk0_distances = np.partition(distances, k0, axis=-1)[..., :k0]  # Shape: (batch_size, n_res, k)
        topk1_distances = np.partition(distances, k1, axis=-1)[..., :k1]  # Shape: (batch_size, n_res, k)
        topk2_distances = np.partition(distances, k2, axis=-1)[..., :k2]  # Shape: (batch_size, n_res, k)

        # Compute the average of the 8 smallest distances
        k0_avg_distances = topk0_distances.mean(axis=-1)  # Shape: (batch_size, n_res)
        k1_avg_distances = topk1_distances.mean(axis=-1)  # Shape: (batch_size, n_res)
        k2_avg_distances = topk2_distances.mean(axis=-1)  # Shape: (batch_size, n_res)

        avg_distances = np.concatenate([k0_avg_distances[...,None], k1_avg_distances[...,None], k2_avg_distances[...,None]], axis = -1)

        return avg_distances

    
################# end of multimer #################
    
