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
from functools import partial
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from triflow.multimer_model.primitives import Linear

from triflow.multimer_model.embedders import (
    RelPosEmbedderMultimer,
    TimeEmbedding,
    SingleOutEmbedder,
    TriangleBlock
)

from triflow.multimer_model.heads import DistogramHead
from triflow.multimer_model.small_hybrid_module import HybridStack 
from triflow.multimer_model.hybrid_structure_module import StructureModule

import triflow.np.residue_constants as residue_constants

from triflow.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)
from triflow.utils.rigid_utils import Rotation, Rigid


class TriFold(nn.Module):
    """
    TriFold.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(TriFold, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template

        # Main trunk + structure module
      
        self.linear_tf_s = Linear(160, self.config["time_embedder"]["c_s"])    
        
        self.time_emb_cat = TimeEmbedding(256, self.config["time_embedder"]["c_s"])

        self.relpos = RelPosEmbedderMultimer(
            **self.config["relpos_embedder"],
        )

        self.triangle_embedder = TriangleBlock(
            **self.config["hybrid_stack"]
        )
       
        self.hybrid = HybridStack(
            **self.config["hybrid_stack"],
        )

        self.distogram_head = DistogramHead(
            self.config["relpos_embedder"]["c_z"], 
            960
        )

        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
    
        self.single_out_embedder = SingleOutEmbedder(self.config["out_embedder"]["tf_dim"], self.config["out_embedder"]["c_s"], self.config["out_embedder"]["c_s"]//2)

    def iteration(self, feats,rigid_frames = None, seq=None):
        # Primary output dictionary
        outputs = {}
        

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype        
        for k in feats:
            if(feats[k].dtype == torch.float32):
                feats[k] = feats[k].to(dtype=dtype)
        if rigid_frames != None:
            rigid_frames = rigid_frames.to(dtype=dtype)

       # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        
        # Prep some features
        batch_dims = feats["residue_index"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n_res = feats["residue_index"].shape[-1]
        device = feats["residue_index"].device

        batch_size = feats["residue_index"].shape[-2]
        
        seq_mask = feats["seq_mask"]
        diffuse_mask = feats["diffuse_mask"]

        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

        # extract burial information
        avg_dist = feats['avg_distances']
        noise_label = feats['noise_label']

        cat_t = feats['cat_t']          
        
        ## Initialize the MSA and pair representations

        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]


        def _dihedrals(X, eps=1e-7):
            """
            Compute dihedral angles from a set of coordinates.
            """
            X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
            dX = X[:, 1:, :] - X[:, :-1, :]
            U = F.normalize(dX, dim=-1)
            u_2 = U[:, :-2, :]
            u_1 = U[:, 1:-1, :]
            u_0 = U[:, 2:, :]
            n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
            n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
            cosD = (n_2 * n_1).sum(-1)
            cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
            D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
            D = F.pad(D, (1, 2), 'constant', 0)
            D = D.view((D.size(0), int(D.size(1) / 3), 3))
            phi, psi, omega = torch.unbind(D, -1)
            D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
            return D_features  # (B, L, 6), the 6 is cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)
        
    
        tf = _dihedrals(torch.nan_to_num(feats["all_atom_positions"])).to(device).float()
        
        #lets run std embedding
        # embed_std = self.time_emb_std(noise_label)[:,None,:]        
        
        embed_cat_t = self.time_emb_cat(cat_t)[:,None,:]
        
        tf = torch.cat([
                        tf,
                        seq, avg_dist,
                        diffuse_mask[...,None],
                        noise_label[...,None, None].expand(-1, n_res, -1),
                        embed_cat_t.expand(-1, n_res, -1)],
                        dim=-1)


        s = self.linear_tf_s(tf)

        z = self.relpos(
            feats,
            inplace_safe=inplace_safe,
            is_training=self.training
        )

        z = self.triangle_embedder(z, pair_mask.to(dtype=z.dtype), inplace_safe=inplace_safe)

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          

        s, z, rigid_frames = self.hybrid(
            s.clone(),
            z,
            rigid_frames,
            seq_mask=seq_mask.to(dtype=s.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            chunk_size=self.globals.chunk_size,
            use_lma=self.globals.use_lma,
            use_flash=self.globals.use_flash,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans,
        )
        outputs["pair"] = z
        outputs["single"] = s
        outputs["hybrid_frames"] = rigid_frames
        outputs["hybrid_logits"] = self.single_out_embedder(outputs["single"])


        outputs["dgram"] = self.distogram_head(z)

        del s, z, rigid_frames

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            rigids = Rigid.from_tensor_7(outputs["hybrid_frames"]),
            mask=feats["seq_mask"].to(dtype=outputs["single"].dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )


        outputs["final_atom_positions"] = outputs["sm"]["frames"][-1][...,-3:]
        outputs["final_atom_mask"] = feats["backbone_rigid_mask"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        outputs["all_logits"] = torch.cat((outputs["hybrid_logits"][None], outputs["sm"]["logits"]), dim = 0)        

        # temperature = 0.01        
        # temperature = 0.1
        # pred_logits = outputs["all_logits"][-1] / temperature
        # probabilities = torch.nn.functional.softmax(pred_logits, dim=-1)
        # pred_aatypes = torch.multinomial(probabilities[0], num_samples=1).squeeze(dim=-1)[None]


        # pred_logits = outputs["all_logits"][-1]
        # pred_aatypes = torch.argmax(pred_logits, dim = -1)



        # outputs["pred_logits"] = pred_logits
        # outputs["pred_aatypes"] = pred_aatypes


        return outputs 

    def forward(self, batch, rigid_frames = None,seq=None):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
        """
        is_grad_enabled = torch.is_grad_enabled()


        # Select the features for the current recycling cycle
        fetch_cur_batch = lambda t: t[..., -1]
        feats = tensor_tree_map(fetch_cur_batch, batch)

        # Enable grad iff we're training and it's the final recycling layer
        with torch.set_grad_enabled(is_grad_enabled):
            # Sidestep AMP bug (PyTorch issue #65766)
            if torch.is_autocast_enabled():
                torch.clear_autocast_cache()

            # Run the next iteration of the model
            outputs = self.iteration(
                feats,
                rigid_frames=rigid_frames,
                seq=seq
            )
        #Run auxiliary heads
#        outputs.update(self.aux_heads(outputs))
        return outputs
    


    def sampler(self, batch, rigid_frames = None,seq=None):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
        """
        is_grad_enabled = torch.is_grad_enabled()


        # Select the features for the current recycling cycle
        # fetch_cur_batch = lambda t: t[..., -1]
        # feats = tensor_tree_map(fetch_cur_batch, batch)

        # Enable grad iff we're training and it's the final recycling layer
        with torch.set_grad_enabled(is_grad_enabled):
            # Sidestep AMP bug (PyTorch issue #65766)
            if torch.is_autocast_enabled():
                torch.clear_autocast_cache()

            # Run the next iteration of the model

            z = self.run_embedder(batch)
            outputs = self.run_blocks(batch, z, rigid_frames, seq)


        return outputs

    def run_embedder(self, feats):

        is_grad_enabled = torch.is_grad_enabled()

        # Enable grad iff we're training and it's the final recycling layer
        with torch.set_grad_enabled(is_grad_enabled):
            # Sidestep AMP bug (PyTorch issue #65766)
            if torch.is_autocast_enabled():
                torch.clear_autocast_cache()


        fetch_cur_batch = lambda t: t[..., -1]
        feats = tensor_tree_map(fetch_cur_batch, feats)
        

        # dtype = next(self.parameters()).dtype
        # for k in feats:
        #     if(feats[k].dtype == torch.float32):
        #         feats[k] = feats[k].to(dtype=dtype)

        

        inplace_safe = not (self.training or torch.is_grad_enabled())
        # Prep some features
        
        batch_dims = feats["residue_index"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n_res = feats["residue_index"].shape[-1]
        device = feats["residue_index"].device

        batch_size = feats["residue_index"].shape[-2]
        
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        
        z = self.relpos(
            feats,
            inplace_safe=inplace_safe,
            is_training=self.training
        )            

        z = self.triangle_embedder(z, pair_mask.to(dtype=z.dtype), inplace_safe=inplace_safe)

        return z



    def run_blocks(self, feats,z, rigid_frames=None, seq=None, temp=0.1):

            z_local = z.clone()

            is_grad_enabled = torch.is_grad_enabled()


            # Select the features for the current recycling cycle
            # fetch_cur_batch = lambda t: t[..., -1]
            # feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            with torch.set_grad_enabled(is_grad_enabled):
                # Sidestep AMP bug (PyTorch issue #65766)
                if torch.is_autocast_enabled():
                    torch.clear_autocast_cache()


            outputs = {}
            fetch_cur_batch = lambda t: t[..., -1]
            feats = tensor_tree_map(fetch_cur_batch, feats)


            dtype = next(self.parameters()).dtype
            # for k in feats:
            #     if(feats[k].dtype == torch.float32):
            #         feats[k] = feats[k].to(dtype=dtype)
            # if rigid_frames != None:
            #     rigid_frames = rigid_frames.to(dtype=dtype)

        # Controls whether the model uses in-place operations throughout
            # The dual condition accounts for activation checkpoints
            inplace_safe = not (self.training or torch.is_grad_enabled())
            # Prep some features
            batch_dims = feats["residue_index"].shape[:-2]
            no_batch_dims = len(batch_dims)
            n_res = feats["residue_index"].shape[-1]
            device = feats["residue_index"].device

            batch_size = feats["residue_index"].shape[-2]
            
            seq_mask = feats["seq_mask"]
            diffuse_mask = feats["diffuse_mask"]
            pair_mask = seq_mask[..., None] * seq_mask[..., None, :]     

            # extract burial information
            avg_dist = feats['avg_distances']
            noise_label = feats['noise_label']

            cat_t = feats['cat_t'] 


            def _dihedrals(X, eps=1e-7):
                X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
                dX = X[:, 1:, :] - X[:, :-1, :]
                U = F.normalize(dX, dim=-1)
                u_2 = U[:, :-2, :]
                u_1 = U[:, 1:-1, :]
                u_0 = U[:, 2:, :]
                n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
                n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)
                cosD = (n_2 * n_1).sum(-1)
                cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
                D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
                D = F.pad(D, (1, 2), 'constant', 0)
                D = D.view((D.size(0), int(D.size(1) / 3), 3))
                phi, psi, omega = torch.unbind(D, -1)
                D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
                return D_features  
                    
            tf = _dihedrals(torch.nan_to_num(feats["all_atom_positions"])).to(device)

            embed_cat_t = self.time_emb_cat(cat_t)[:,None,:]

            
            tf = torch.cat([
                            tf,
                            seq, avg_dist,
                            diffuse_mask[...,None],
                            noise_label[...,None, None].expand(-1, n_res, -1),
                            embed_cat_t.expand(-1, n_res, -1)],
                            dim=-1)
            
            

            s = self.linear_tf_s(tf)

            s, z_local, rigid_frames = self.hybrid(
                s.clone(),
                z_local,
                rigid_frames,
                seq_mask=seq_mask.to(dtype=s.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )
            outputs["pair"] = z_local
            outputs["single"] = s
            outputs["hybrid_frames"] = rigid_frames
            outputs["hybrid_logits"] = self.single_out_embedder(outputs["single"])

            # outputs["dgram"] = self.distogram_head(z)

            del s, z_local, rigid_frames

            # Predict 3D structure
            outputs["sm"] = self.structure_module(
                outputs,
                rigids = Rigid.from_tensor_7(outputs["hybrid_frames"]),
                mask=feats["seq_mask"].to(dtype=outputs["single"].dtype),
                inplace_safe=inplace_safe,
                _offload_inference=self.globals.offload_inference,
            )

            outputs["final_atom_positions"] = outputs["sm"]["frames"][-1][...,-3:]
            outputs["final_atom_mask"] = feats["backbone_rigid_mask"]
            outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

            outputs["all_logits"] = torch.cat((outputs["hybrid_logits"][None], outputs["sm"]["logits"]), dim = 0)
            
            pred_logits = outputs["all_logits"][-1] 
            outputs["pred_logits"] = pred_logits
            
            return outputs 


















