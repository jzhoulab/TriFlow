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
import pandas as pd

import torch.nn as nn

from triflow.data import templates, parsers, mmcif_parsing
from triflow.data.templates import get_custom_template_features
from triflow.data.tools import jackhmmer, hhblits, hhsearch
from triflow.data.tools.utils import to_date 
from triflow.np import residue_constants, protein
from triflow.np import multimer_residue_constants as rc


FeatureDict = Mapping[str, np.ndarray]



def make_pdb_features_multimer(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 80.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    # pdb_feats["asym_id"] = protein_object.chain_index
    pdb_feats["noiseless_all_atom_positions"] = pdb_feats["all_atom_positions"]

    if(is_distillation):        
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def empty_template_feats(n_res) -> FeatureDict:
    return {
        "template_aatype": np.zeros((0, n_res)).astype(np.int64),
        "template_all_atom_positions": 
            np.zeros((0, n_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, n_res, 37)).astype(np.float32),
    }


def make_template_features(
    input_sequence: str,
    hits: Sequence[Any],
    template_featurizer: Any,
    query_pdb_code: Optional[str] = None,
    query_release_date: Optional[str] = None,
) -> FeatureDict:
    hits_cat = sum(hits.values(), [])
    if(len(hits_cat) == 0 or template_featurizer is None):
        template_features = empty_template_feats(len(input_sequence))
    else:
        templates_result = template_featurizer.get_templates(
            query_sequence=input_sequence,
            query_pdb_code=query_pdb_code,
            query_release_date=query_release_date,
            hits=hits_cat,
        )
        template_features = templates_result.features

        # The template featurizer doesn't format empty template features
        # properly. This is a quick fix.
        if(template_features["template_aatype"].shape[0] == 0):
            template_features = empty_template_feats(len(input_sequence))

    return template_features


def unify_template_features(
    template_feature_list: Sequence[FeatureDict]
) -> FeatureDict:
    out_dicts = []
    seq_lens = [fd["template_aatype"].shape[1] for fd in template_feature_list]
    for i, fd in enumerate(template_feature_list):
        out_dict = {}
        n_templates, n_res = fd["template_aatype"].shape[:2]
        for k,v in fd.items():
            seq_keys = [
                "template_aatype",
                "template_all_atom_positions",
                "template_all_atom_mask",
            ]
            if(k in seq_keys):
                new_shape = list(v.shape)
                assert(new_shape[1] == n_res)
                new_shape[1] = sum(seq_lens)
                new_array = np.zeros(new_shape, dtype=v.dtype)
                
                if(k == "template_aatype"):
                    new_array[..., residue_constants.HHBLITS_AA_TO_ID['-']] = 1

                offset = sum(seq_lens[:i])
                new_array[:, offset:offset + seq_lens[i]] = v
                out_dict[k] = new_array
            else:
                out_dict[k] = v

        chain_indices = np.array(n_templates * [i])
        out_dict["template_chain_index"] = chain_indices

        if(n_templates != 0):
            out_dicts.append(out_dict)

    if(len(out_dicts) > 0):
        out_dict = {
            k: np.concatenate([od[k] for od in out_dicts]) for k in out_dicts[0]
        }
    else:
        out_dict = empty_template_feats(sum(seq_lens))

    return out_dict


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
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features


def make_mmcif_features(
    mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_
    )

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])

def _aatype_with_dom_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x_dom[aatype[i]]
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
    confidence_threshold: float = 80.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if(is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix],
) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f"MSA {msa_index} must contain at least one sequence."
            )
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
            )
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array(
        [num_alignments] * num_res, dtype=np.int32
    )
    return features


def make_sequence_features_with_custom_template(
        sequence: str,
        mmcif_path: str,
        pdb_id: str,
        chain_id: str,
        kalign_binary_path: str) -> FeatureDict:
    """
    process a single fasta file using features derived from a single template rather than an alignment
    """
    num_res = len(sequence)

    sequence_features = make_sequence_features(
        sequence=sequence,
        description=pdb_id,
        num_res=num_res,
    )

    msa_data = [[sequence]]
    deletion_matrix = [[[0 for _ in sequence]]]

    msa_features = make_msa_features(msa_data, deletion_matrix)
    template_features = get_custom_template_features(
        mmcif_path=mmcif_path,
        query_sequence=sequence,
        pdb_id=pdb_id,
        chain_id=chain_id,
        kalign_binary_path=kalign_binary_path
    )

    return {
        **sequence_features,
        **msa_features,
        **template_features.features
    }

class AlignmentRunner:
    """Runs alignment tools and saves the results"""
    def __init__(
        self,
        jackhmmer_binary_path: Optional[str] = None,
        hhblits_binary_path: Optional[str] = None,
        hhsearch_binary_path: Optional[str] = None,
        uniref90_database_path: Optional[str] = None,
        mgnify_database_path: Optional[str] = None,
        bfd_database_path: Optional[str] = None,
        uniclust30_database_path: Optional[str] = None,
        pdb70_database_path: Optional[str] = None,
        use_small_bfd: Optional[bool] = None,
        no_cpus: Optional[int] = None,
        uniref_max_hits: int = 10000,
        mgnify_max_hits: int = 5000,
    ):
        """
        Args:
            jackhmmer_binary_path:
                Path to jackhmmer binary
            hhblits_binary_path:
                Path to hhblits binary
            hhsearch_binary_path:
                Path to hhsearch binary
            uniref90_database_path:
                Path to uniref90 database. If provided, jackhmmer_binary_path
                must also be provided
            mgnify_database_path:
                Path to mgnify database. If provided, jackhmmer_binary_path
                must also be provided
            bfd_database_path:
                Path to BFD database. Depending on the value of use_small_bfd,
                one of hhblits_binary_path or jackhmmer_binary_path must be 
                provided.
            uniclust30_database_path:
                Path to uniclust30. Searched alongside BFD if use_small_bfd is 
                false.
            pdb70_database_path:
                Path to pdb70 database.
            use_small_bfd:
                Whether to search the BFD database alone with jackhmmer or 
                in conjunction with uniclust30 with hhblits.
            no_cpus:
                The number of CPUs available for alignment. By default, all
                CPUs are used.
            uniref_max_hits:
                Max number of uniref hits
            mgnify_max_hits:
                Max number of mgnify hits
        """
        db_map = {
            "jackhmmer": {
                "binary": jackhmmer_binary_path,
                "dbs": [
                    uniref90_database_path,
                    mgnify_database_path,
                    bfd_database_path if use_small_bfd else None,
                ],
            },
            "hhblits": {
                "binary": hhblits_binary_path,
                "dbs": [
                    bfd_database_path if not use_small_bfd else None,
                ],
            },
            "hhsearch": {
                "binary": hhsearch_binary_path,
                "dbs": [
                    pdb70_database_path,
                ],
            },
        }

        for name, dic in db_map.items():
            binary, dbs = dic["binary"], dic["dbs"]
            if(binary is None and not all([x is None for x in dbs])):
                raise ValueError(
                    f"{name} DBs provided but {name} binary is None"
                )

        if(not all([x is None for x in db_map["hhsearch"]["dbs"]])
            and uniref90_database_path is None):
            raise ValueError(
                """uniref90_database_path must be specified in order to perform
                   template search"""
            )

        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits
        self.use_small_bfd = use_small_bfd

        if(no_cpus is None):
            no_cpus = cpu_count()

        self.jackhmmer_uniref90_runner = None
        if(jackhmmer_binary_path is not None and 
            uniref90_database_path is not None
        ):
            self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniref90_database_path,
                n_cpu=no_cpus,
            )
   
        self.jackhmmer_small_bfd_runner = None
        self.hhblits_bfd_uniclust_runner = None
        if(bfd_database_path is not None):
            if use_small_bfd:
                self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=bfd_database_path,
                    n_cpu=no_cpus,
                )
            else:
                dbs = [bfd_database_path]
                if(uniclust30_database_path is not None):
                    dbs.append(uniclust30_database_path)
                self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=dbs,
                    n_cpu=no_cpus,
                )

        self.jackhmmer_mgnify_runner = None
        if(mgnify_database_path is not None):
            self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=mgnify_database_path,
                n_cpu=no_cpus,
            )

        self.hhsearch_pdb70_runner = None
        if(pdb70_database_path is not None):
            self.hhsearch_pdb70_runner = hhsearch.HHSearch(
                binary_path=hhsearch_binary_path,
                databases=[pdb70_database_path],
                n_cpu=no_cpus,
            )

    def run(
        self,
        fasta_path: str,
        output_dir: str,
    ):
        """Runs alignment tools on a sequence"""
        if(self.jackhmmer_uniref90_runner is not None):
            jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
                fasta_path
            )[0]
            uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
                jackhmmer_uniref90_result["sto"], 
                max_sequences=self.uniref_max_hits
            )
            uniref90_out_path = os.path.join(output_dir, "uniref90_hits.a3m")
            with open(uniref90_out_path, "w") as f:
                f.write(uniref90_msa_as_a3m)

            if(self.hhsearch_pdb70_runner is not None):
                hhsearch_result = self.hhsearch_pdb70_runner.query(
                    uniref90_msa_as_a3m
                )
                pdb70_out_path = os.path.join(output_dir, "pdb70_hits.hhr")
                with open(pdb70_out_path, "w") as f:
                    f.write(hhsearch_result)

        if(self.jackhmmer_mgnify_runner is not None):
            jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
                fasta_path
            )[0]
            mgnify_msa_as_a3m = parsers.convert_stockholm_to_a3m(
                jackhmmer_mgnify_result["sto"], 
                max_sequences=self.mgnify_max_hits
            )
            mgnify_out_path = os.path.join(output_dir, "mgnify_hits.a3m")
            with open(mgnify_out_path, "w") as f:
                f.write(mgnify_msa_as_a3m)

        if(self.use_small_bfd and self.jackhmmer_small_bfd_runner is not None):
            jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(
                fasta_path
            )[0]
            bfd_out_path = os.path.join(output_dir, "small_bfd_hits.sto")
            with open(bfd_out_path, "w") as f:
                f.write(jackhmmer_small_bfd_result["sto"])
        elif(self.hhblits_bfd_uniclust_runner is not None):
            hhblits_bfd_uniclust_result = (
                self.hhblits_bfd_uniclust_runner.query(fasta_path)
            )
            if output_dir is not None:
                bfd_out_path = os.path.join(output_dir, "bfd_uniclust_hits.a3m")
                with open(bfd_out_path, "w") as f:
                    f.write(hhblits_bfd_uniclust_result["a3m"])


class DataPipeline:
    """Assembles input features."""
#    def __init__(
#        self,
#        template_featurizer: Optional[templates.TemplateHitFeaturizer],
#    ):
#        self.template_featurizer = template_featurizer

    def _parse_msa_data(
        self,
        alignment_dir: str,
        alignment_index: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        msa_data = {} 
        if(alignment_index is not None):
            fp = open(os.path.join(alignment_dir, alignment_index["db"]), "rb")

            def read_msa(start, size):
                fp.seek(start)
                msa = fp.read(size).decode("utf-8")
                return msa

            for (name, start, size) in alignment_index["files"]:
                ext = os.path.splitext(name)[-1]

                if(ext == ".a3m"):
                    msa, deletion_matrix = parsers.parse_a3m(
                        read_msa(start, size)
                    )
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                elif(ext == ".sto"):
                    msa, deletion_matrix, _ = parsers.parse_stockholm(
                        read_msa(start, size)
                    )
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                else:
                    continue
               
                msa_data[name] = data
            
            fp.close()
        else: 
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if(ext == ".a3m"):
                    with open(path, "r") as fp:
                        msa, deletion_matrix = parsers.parse_a3m(fp.read())
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                elif(ext == ".sto"):
                    with open(path, "r") as fp:
                        msa, deletion_matrix, _ = parsers.parse_stockholm(
                            fp.read()
                        )
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                else:
                    continue
                
                msa_data[f] = data

        return msa_data

    def _parse_template_hits(
        self,
        alignment_dir: str,
        alignment_index: Optional[Any] = None
    ) -> Mapping[str, Any]:
        all_hits = {}
        if(alignment_index is not None):
            fp = open(os.path.join(alignment_dir, alignment_index["db"]), 'rb')

            def read_template(start, size):
                fp.seek(start)
                return fp.read(size).decode("utf-8")

            for (name, start, size) in alignment_index["files"]:
                ext = os.path.splitext(name)[-1]

                if(ext == ".hhr"):
                    hits = parsers.parse_hhr(read_template(start, size))
                    all_hits[name] = hits

            fp.close()
        else:
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if(ext == ".hhr"):
                    with open(path, "r") as fp:
                        hits = parsers.parse_hhr(fp.read())
                    all_hits[f] = hits

        return all_hits

    def _get_msas(self,
        alignment_dir: str,
        input_sequence: Optional[str] = None,
        alignment_index: Optional[str] = None,
    ):
        msa_data = self._parse_msa_data(alignment_dir, alignment_index)
        if(len(msa_data) == 0):
            if(input_sequence is None):
                raise ValueError(
                    """
                    If the alignment dir contains no MSAs, an input sequence 
                    must be provided.
                    """
                )
            msa_data["dummy"] = {
                "msa": [input_sequence],
                "deletion_matrix": [[0 for _ in input_sequence]],
            }

        msas, deletion_matrices = zip(*[
            (v["msa"], v["deletion_matrix"]) for v in msa_data.values()
        ])

        return msas, deletion_matrices

    def _process_msa_feats(
        self,
        alignment_dir: str,
        input_sequence: Optional[str] = None,
        alignment_index: Optional[str] = None
    ) -> Mapping[str, Any]:
        msas, deletion_matrices = self._get_msas(
            alignment_dir, input_sequence, alignment_index
        )
        msa_features = make_msa_features(
            msas=msas,
            deletion_matrices=deletion_matrices,
        )

        return msa_features

    def process_fasta(
        self,
        fasta_path: str,
        alignment_dir: str,
        alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """Assembles features for a single sequence in a FASTA file""" 
        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f"More than one input sequence found in {fasta_path}."
            )
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)
        
        return {
            **sequence_features,
            **msa_features, 
            **template_features
        }

    def process_mmcif(
        self,
        mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
#        alignment_dir: str,
        chain_id: Optional[str] = None,
#        alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a specific chain in an mmCIF object.

            If chain_id is None, it is assumed that there is only one chain
            in the object. Otherwise, a ValueError is thrown.
        """
        if chain_id is None:
            chains = mmcif.structure.get_chains()
            chain = next(chains, None)
            if chain is None:
                raise ValueError("No chains in mmCIF file")
            chain_id = chain.id

        mmcif_feats = make_mmcif_features(mmcif, chain_id)

        input_sequence = mmcif.chain_to_seqres[chain_id]
#        hits = self._parse_template_hits(alignment_dir, alignment_index)
#        template_features = make_template_features(
#            input_sequence,
#            hits,
#            self.template_featurizer,
#            query_release_date=to_date(mmcif.header["release_date"])
#        )
        
#        msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)
        return {**mmcif_feats}

    def process_pdb(
        self,
        pdb_path: str,
#        alignment_dir: str,
        is_distillation: bool = True,
        chain_id: Optional[str] = None,
        _structure_index: Optional[str] = None,
#        alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file

        """
        if(_structure_index is not None):
            db_dir = os.path.dirname(pdb_path)
            db = _structure_index["db"]
            db_path = os.path.join(db_dir, db)
            fp = open(db_path, "rb")
            _, offset, length = _structure_index["files"][0]
            fp.seek(offset)
            pdb_str = fp.read(length).decode("utf-8")
            fp.close()
        else:
            with open(pdb_path, 'r') as f:
                pdb_str = f.read()

        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype) 
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features(
            protein_object, 
            description, 
            is_distillation=is_distillation
        )

#        hits = self._parse_template_hits(alignment_dir, alignment_index)
#        template_features = make_template_features(
#            input_sequence,
#            hits,
#            self.template_featurizer,
#        )

#        msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {**pdb_feats}
    


    def process_cath_cache(self, protein):

        features= {}

        #process all the sequence features
        sequence = protein["seq"]
        features["aatype"] = residue_constants.sequence_to_onehot(
            sequence=sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )
        
        num_res = len(sequence)
        features["residue_index"] = np.array(range(num_res), dtype=np.int32)
        features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
        features["sequence"] = np.array( [sequence.encode("utf-8")], dtype=object)


        #coordinate information

        features["all_atom_positions"] = protein["coords"].astype(np.float32)
        features["all_atom_mask"] = protein["all_atom_mask"].astype(np.float32)
        features["resolution"] = np.array([0.]).astype(np.float32)
        features["is_distillation"] = np.array(0.).astype(np.float32)


        #random stuff
        features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
        features["domain_name"] = np.array([protein["CATH"][0].encode("utf-8")], dtype=object)

        return features
    

   
        
################# start of qc multimer #################

    def process_pdb_multimer(self, 
                            pdb_path: str,
                            is_distillation: bool = False,
                            chain_id: Optional[str] = None,                              
                            ):
        with open(pdb_path, 'r') as f:
            pdb_string = f.read()
            
        protein_object = protein.from_pdb_string(pdb_string, chain_id=None)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features_multimer(
            protein_object, 
            description, 
            is_distillation=is_distillation
        )

        pdb_feats["backbone_rigid_mask"] = pdb_feats["all_atom_mask"][...,1] #pick ca mask
        
        pdb_feats["seq_mask"] = np.ones(pdb_feats["backbone_rigid_mask"].shape[0])
        
        return pdb_feats


    def get_multimer_path(self, csv_row, source):

        ddi_base_path = "/archive/bioinformatics/Zhou_lab/shared/yuanrq/data/AFDB_MULTIDOM/dompdbs/net/tukwila/congq/AFDB_multidom/dompdbs"
        ppi_base_path = "/archive/bioinformatics/Zhou_lab/shared/yuanrq/data/PDB_COMPLEX_QC/posi_pdbs/net/tukwila/congq/PDB_complex/posi_pdbs"
        
        ddi = source == "ddi"
        pair_parts = csv_row.Domain_pair.split(':')

        pair1 = pair_parts[0]
        pair2 = pair_parts[1]

        hash_code = csv_row.Hash

        if ddi:
            pdb_path1 = ddi_base_path + '/' + hash_code + '/' + pair1 + '.pdb'
            pdb_path2 = ddi_base_path + '/' + hash_code + '/' + pair2 + '.pdb'        

        else:
            #assuming using the ppi dataset
            pdb_path1 = ppi_base_path + '/' + hash_code + '/' + pair1 + '-' + pair2 + "__" + pair1  + '.pdb'
            pdb_path2 = ppi_base_path + '/' + hash_code + '/' + pair1 + '-' + pair2 + "__" + pair2  + '.pdb'

        return [pdb_path1, pdb_path2]
    

    # def get_monomer_path(self, csv_row, ddi=False):

    #     ddi_base_path = "/archive/bioinformatics/Zhou_lab/shared/yuanrq/data/AFDB_MULTIDOM/dompdbs/net/tukwila/congq/AFDB_multidom/dompdbs"
    #     ppi_base_path = "/archive/bioinformatics/Zhou_lab/shared/yuanrq/data/PDB_COMPLEX_QC/posi_pdbs/net/tukwila/congq/PDB_complex/posi_pdbs"
        
    #     pair_parts = csv_row.Domain_pair.split(':')

    #     pair1 = pair_parts[0]
    #     pair2 = pair_parts[1]

    #     chain = csv_row.split1
    #     hash_code  = csv_row.Hash
    #     if ddi:
    #         pdb_path1 = ddi_base_path + '/' + hash_code + '/' + chain + '.pdb'
            
          
    #     else:
    #         #assuming using the ppi dataset
    #         pdb_path1 = ppi_base_path + '/' + hash_code + '/' + pair1 + '-' + pair2 + "__" + chain  + '.pdb'
            

    #     return [pdb_path1]
    

    def get_monomer_path(self, csv_row, source):

        pdb_base_path = "/archive/bioinformatics/Zhou_lab/shared/yuanrq/data/domain/random_sample/PDB_domains/all/"
        afdb_base_path = "/archive/bioinformatics/Zhou_lab/shared/yuanrq/data/domain/AFDB_domains_processed_split_2/train"
        afdb_val_base_path = "/archive/bioinformatics/Zhou_lab/shared/yuanrq/data/domain/AFDB_domains_processed_split_2/val"        


        file_id = str(csv_row['id'])
        
        if source == "afdb":
            sub_dir_2 = file_id[-2:]
            sub_dir_1 = file_id[-4:-2]
            pdb_path = os.path.join(afdb_base_path,sub_dir_1, sub_dir_2, file_id) + ".pdb"

        if source == "pdb":
            pdb_path = os.path.join(pdb_base_path, file_id) + ".pdb"

        if source == "valid":
            sub_dir_2 = file_id[-2:]
            sub_dir_1 = file_id[-4:-2]
            pdb_path = os.path.join(afdb_val_base_path, sub_dir_1, sub_dir_2, file_id) + ".pdb"


        return [pdb_path]    
          
    

    def process_multimer_chains(self, item, mode,):



        source = item.get('Source', None)
        # is_ddi = item.get('Source', None) == 'ddi'
        is_monomer = item.get("datatype", None) == "monomer"

        is_distillation = source == "ddi" or source == "afdb"

        if not is_monomer:
            pdb_paths = self.get_multimer_path(item, source)
        else:
            pdb_paths = self.get_monomer_path(item, source)

            
        ###################

        features= {
            'aatype': [],
            'between_segment_residues': [],
            'residue_index': [],
            'seq_length': [],
            'sequence': [],
            'all_atom_positions': [],
            'all_atom_mask': [],
            'resolution': [],
            'asym_id': [],
            'noiseless_all_atom_positions': [],
            'backbone_rigid_mask': [],
            'seq_mask': [],
        }        

        concatenated_seq = ""


        chain_idx_counter = 0
        for pdb_path in pdb_paths:
            with open(pdb_path, 'r') as f:
                pdb_string = f.read()
                
            protein_object = protein.from_pdb_string(pdb_string, chain_id=None)
            input_sequence = _aatype_to_str_sequence(protein_object.aatype)
            description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
            pdb_feats = make_pdb_features_multimer(
                protein_object, 
                description, 
                is_distillation=is_distillation
            )

            
            pdb_feats["backbone_rigid_mask"] = pdb_feats["all_atom_mask"][...,1] #pick ca mask            
            pdb_feats["seq_mask"] = np.ones(pdb_feats["backbone_rigid_mask"].shape[0])
            pdb_feats["seq"] = np.array(input_sequence)

                        
            for key in features.keys():                
                if key in pdb_feats.keys():
                    features[key].append(pdb_feats[key])  
            

            features["asym_id"].append(np.full((len(pdb_feats["seq_mask"]),), chain_idx_counter))

            chain_idx_counter = chain_idx_counter + 1                        
            concatenated_seq += input_sequence

            # Concatenate arrays along the first dimension (axis 0)
        for key in features.keys():
            if isinstance(features[key], list) and len(features[key]) > 0:
                features[key] = np.concatenate(features[key], axis=0)                 

        features['seq'] = concatenated_seq

        ###################

        num_res = features["seq_mask"].shape[0]
        features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32) #need to rewrite seq_length

        #mask seqs where positions do not exist, his tag and X res
        res_mask = np.zeros_like(features["backbone_rigid_mask"]).astype(np.float32)

        for idx in list(np.unique(features['asym_id'])):
            res = np.argwhere(features['asym_id'] == idx)[None,:,0]
            initial_sequence= "".join(list(np.array(list(features['seq']))[res][0,]))
            # initial_sequence= "".join(list(np.array(list(features['seq']))[res][:,0])) 
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
            # res_mask[res[:,0]] = 1

        res_mask = res_mask * features["backbone_rigid_mask"]  
        unk_pos = np.where(features["aatype"][:, -1] == 1)[0] #positions with unknown residues
        res_mask[unk_pos] = 0
        
        features["seq_mask"] = res_mask.astype(np.int32) 
        features["noiseless_all_atom_positions"] = features["all_atom_positions"]
        features["noise_label"] = np.array(0.).astype(np.float32)



        to_noise = torch.rand(1) < 0.5
        #add a little bit of nosie to af structures to avoid overfitting         
        if is_distillation and not to_noise:
            features["all_atom_positions"] = features["all_atom_positions"] + ( np.random.randn(*features["all_atom_positions"].shape) * 0.02)
        
        #lets add some gaussian noise to the atom positions 
        if mode == 'train':
            if to_noise:
            # if is_distillation:            
                
                z = np.random.randn(*features["all_atom_positions"].shape)                
                std = 0.2                 
                features["all_atom_positions"] = features["all_atom_positions"] + (z * std)
                features["noise_label"] =  np.array(1.).astype(np.float32)

        
        features["resolution"] = np.array([0.]).astype(np.float32)
        features["is_distillation"] = np.array(0.).astype(np.float32)

        #lets also get packing information
        
        # features['avg_distances'] = np.nan_to_num(self.compute_avg_distance(features['noiseless_all_atom_positions'][:,1,:][None]))[0]  #calculate closest atoms based on Ca distances        
        features['avg_distances'] = np.nan_to_num(self.compute_avg_distance(features['all_atom_positions'][:,1,:][None]))[0].astype(np.float32)  #calculate closest atoms based on Ca distances
        

        #random stuff
        features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
        features["mask_everything"] = np.array(0.).astype(np.float32) #do not mask all the positions in a sequence
                
        # features["is_ddi"] = np.array(int(source))
        features["is_monomer"] = np.array(int(is_monomer))
            
        return features    

    
    def process_bb_multimer_chains(self, item, mode,):



        source = item.get('Source', None)
        # is_ddi = item.get('Source', None) == 'ddi'
        is_monomer = item.get("datatype", None) == "monomer"

        is_distillation = source == "ddi" or source == "afdb"

        if not is_monomer:
            pdb_paths = self.get_multimer_path(item, source)
        else:
            pdb_paths = self.get_monomer_path(item, source)

            
        ###################

        features= {
            'aatype': [],
            'between_segment_residues': [],
            'residue_index': [],
            'seq_length': [],
            'sequence': [],
            'all_atom_positions': [],
            'all_atom_mask': [],
            'resolution': [],
            'asym_id': [],
            'noiseless_all_atom_positions': [],
            'backbone_rigid_mask': [],
            'seq_mask': [],
        }        

        concatenated_seq = ""


        chain_idx_counter = 0
        for pdb_path in pdb_paths:
            with open(pdb_path, 'r') as f:
                pdb_string = f.read()
                
            protein_object = protein.from_pdb_string(pdb_string, chain_id=None)
            input_sequence = _aatype_to_str_sequence(protein_object.aatype)
            description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
            pdb_feats = make_pdb_features_multimer(
                protein_object, 
                description, 
                is_distillation=is_distillation
            )

            
            pdb_feats["backbone_rigid_mask"] = pdb_feats["all_atom_mask"][...,1] #pick ca mask            
            pdb_feats["seq_mask"] = np.ones(pdb_feats["backbone_rigid_mask"].shape[0])
            pdb_feats["seq"] = np.array(input_sequence)

                        
            for key in features.keys():                
                if key in pdb_feats.keys():
                    features[key].append(pdb_feats[key])  
            

            features["asym_id"].append(np.full((len(pdb_feats["seq_mask"]),), chain_idx_counter))

            chain_idx_counter = chain_idx_counter + 1                        
            concatenated_seq += input_sequence

            # Concatenate arrays along the first dimension (axis 0)
        for key in features.keys():
            if isinstance(features[key], list) and len(features[key]) > 0:
                features[key] = np.concatenate(features[key], axis=0)                 

        features['seq'] = concatenated_seq

        ###################

        num_res = features["seq_mask"].shape[0]
        features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32) #need to rewrite seq_length

        #mask seqs where positions do not exist, his tag and X res
        res_mask = np.zeros_like(features["backbone_rigid_mask"]).astype(np.float32)

        for idx in list(np.unique(features['asym_id'])):
            res = np.argwhere(features['asym_id'] == idx)[None,:,0]
            initial_sequence= "".join(list(np.array(list(features['seq']))[res][0,]))
            # initial_sequence= "".join(list(np.array(list(features['seq']))[res][:,0])) 
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
            # res_mask[res[:,0]] = 1

        res_mask = res_mask * features["backbone_rigid_mask"]  
        unk_pos = np.where(features["aatype"][:, -1] == 1)[0] #positions with unknown residues
        res_mask[unk_pos] = 0
        
        features["seq_mask"] = res_mask.astype(np.int32) 
        features["noiseless_all_atom_positions"] = features["all_atom_positions"]                
        
        features["resolution"] = np.array([0.]).astype(np.float32)
        features["is_distillation"] = np.array(0.).astype(np.float32)

        #lets also get packing information
        # features['avg_distances'] = np.nan_to_num(self.compute_avg_distance(features['all_atom_positions'][:,1,:][None]))[0].astype(np.float32)  #calculate closest atoms based on Ca distances
        
        #random stuff
        features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
        features["mask_everything"] = np.array(0.).astype(np.float32) #do not mask all the positions in a sequence
                
        # features["is_ddi"] = np.array(int(source))
        features["is_monomer"] = np.array(int(is_monomer))

        #create diffuse mask
        if is_monomer:
            features["diffuse_mask"] = np.ones((num_res), dtype=np.int32)

        else:
            features["diffuse_mask"] = self._sample_scaffold_mask(num_res, features["asym_id"], res_mask)



    
        return features    





    def _sample_scaffold_mask(self, num_res, chain_idx, res_mask):

        unique_chains = np.unique(chain_idx)
        selected_chain = unique_chains[np.random.randint(len(unique_chains))]
        chain_residues = np.where(chain_idx == selected_chain)[0]
        motif_mask = np.zeros(num_res)
        motif_mask[chain_residues] = 1.0        
        scaffold_mask = 1 - motif_mask

        return scaffold_mask * res_mask


    # def compute_avg_distance(self, positions):
    #     """
    #     Computes the average distance to the 8 closest residues for each residue.

    #     Args:
    #         positions (numpy.ndarray): Array of shape (batch_size, n_res, 3) containing the positions.

    #     Returns:
    #         numpy.ndarray: Array of shape (batch_size, n_res) containing the average distances.
    #     """
    #     batch_size, n_res, _ = positions.shape

    #     # Compute pairwise squared distances
    #     diff = positions[:, :, None, :] - positions[:, None, :, :]  # Shape: (batch_size, n_res, n_res, 3)
    #     squared_distances = np.sum(diff ** 2, axis=-1)  # Shape: (batch_size, n_res, n_res)
    #     distances = np.sqrt(squared_distances + 1e-8)  # Add epsilon to avoid sqrt(0)

    #     # Exclude self-distances by setting diagonals to infinity
    #     mask = np.eye(n_res, dtype=bool)[None, :, :]  # Shape: (1, n_res, n_res)
    #     distances[mask] = np.inf

    #     # Find the 8 smallest distances for each residue
    #     k = 8
    #     if n_res <=8:
    #         k = n_res - 1        
        
    #     topk_distances = np.partition(distances, k, axis=-1)[..., :k]  # Shape: (batch_size, n_res, k)

    #     # Compute the average of the 8 smallest distances

    #     avg_distances = topk_distances.mean(axis=-1)  # Shape: (batch_size, n_res)

    #     return avg_distances

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


################# end of qc multimer #################
    
    def process_core(
        self,
        core_path: str,
        alignment_dir: str,
        alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a ProteinNet .core file.
        """
        with open(core_path, 'r') as f:
            core_str = f.read()

        protein_object = protein.from_proteinnet_string(core_str)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype) 
        description = os.path.splitext(os.path.basename(core_path))[0].upper()
        core_feats = make_protein_features(protein_object, description)
        
        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        msa_features = self._process_msa_feats(alignment_dir, input_sequence)

        return {**core_feats, **template_features, **msa_features}

    def process_multiseq_fasta(self,
        fasta_path: str,
        super_alignment_dir: str,
        ri_gap: int = 200,
    ) -> FeatureDict:
        """
            Assembles features for a multi-sequence FASTA. Uses Minkyung Baek's
            hack from Twitter (a.k.a. AlphaFold-Gap).
        """
        with open(fasta_path, 'r') as f:
            fasta_str = f.read()

        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        
        # No whitespace allowed
        input_descs = [i.split()[0] for i in input_descs]

        # Stitch all of the sequences together
        input_sequence = ''.join(input_seqs)
        input_description = '-'.join(input_descs)
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        seq_lens = [len(s) for s in input_seqs]
        total_offset = 0
        for sl in seq_lens:
            total_offset += sl
            sequence_features["residue_index"][total_offset:] += ri_gap

        msa_list = []
        deletion_mat_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(
                super_alignment_dir, desc
            )
            msas, deletion_mats = self._get_msas(
                alignment_dir, seq, None
            )
            msa_list.append(msas)
            deletion_mat_list.append(deletion_mats) 

        final_msa = []
        final_deletion_mat = []
        msa_it = enumerate(zip(msa_list, deletion_mat_list))
        for i, (msas, deletion_mats) in msa_it:
            prec, post = sum(seq_lens[:i]), sum(seq_lens[i + 1:])
            msas = [
                [prec * '-' + seq + post * '-' for seq in msa] for msa in msas
            ]
            deletion_mats = [
                [prec * [0] + dml + post * [0] for dml in deletion_mat] 
                for deletion_mat in deletion_mats
            ]

            assert(len(msas[0][-1]) == len(input_sequence))

            final_msa.extend(msas)
            final_deletion_mat.extend(deletion_mats)

        msa_features = make_msa_features(
            msas=final_msa,
            deletion_matrices=final_deletion_mat,
        )

        template_feature_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(
                super_alignment_dir, desc
            )
            hits = self._parse_template_hits(alignment_dir, alignment_index=None)
            template_features = make_template_features(
                seq,
                hits,
                self.template_featurizer,
            )
            template_feature_list.append(template_features)

        template_features = unify_template_features(template_feature_list)

        return {
            **sequence_features,
            **msa_features, 
            **template_features,
        }
