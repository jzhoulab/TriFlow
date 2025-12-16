import argparse
import os
import random
import torch
import numpy as np
import time
import json  
from tqdm import tqdm  


import torch.nn as nn
from triflow.utils.rigid_utils import Rigid
from triflow.utils.tensor_utils import tensor_tree_map
from triflow.multimer_config import model_config
from triflow.mpnn_data.data_pipeline import _aatype_to_str_sequence
from triflow.multimer_model.discrete_cond_fm_model import TriFold

from triflow.mpnn_data import data_pipeline, feature_pipeline
from triflow.utils.loss import scale_trans, add_trans_noise
from triflow.np import protein
from triflow.np.residue_constants import restype_order_with_x

from triflow.flow_config import get_config
from triflow.utils.interp import aa_interpolant
from triflow.afdb_data.data_transforms import get_interface_residues


class TriFoldPredictor:
    """
    A class to load a TriFold model and predict sequences from PDB structures.
    """

    def __init__(self, ckpt_path=None, model_config_preset='initial_training', device='cuda:0'):
        """
        Initializes the TriFoldPredictor.

        Args:
            ckpt_path (str, optional): Path to the model checkpoint file. Defaults to the example path if not provided.
            model_config_preset (str): The model configuration preset to use.
            device (str): Device to run the model on (e.g., 'cpu' or 'cuda:0').
        """
        
        self.ckpt_path = ckpt_path or "./weights/afdb_dataset/afdb_weights.pt"
        self.device = device
        self.config = model_config(model_config_preset, train="False")
        self.flow_config = get_config()

        self.model = TriFold(self.config).eval().to(self.device)
        self.model.compile()

        # Load model weights
        self.load_model_weights(self.ckpt_path)
        
        

        # Initialize data pipeline and feature pipeline
        self.data_pipe = data_pipeline.DataPipeline()
        self.feature_pipeline_func = feature_pipeline.FeaturePipeline(self.config.data)

        self.interpolant = aa_interpolant.Interpolant(self.flow_config.interpolant)
        self.interpolant.set_device(self.device)

    def load_model_weights(self, ckpt_path):
        """
        Loads the model weights from a checkpoint file.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(ckpt_path, map_location=self.device)        
        state_dict = checkpoint["ema"]['params']
        
        self.model.load_state_dict(state_dict)
        print(f"Loaded model weights from {ckpt_path}")


    def get_contacts(self, positions, atom_mask, asym_id, contact_threshold=10):

        diff = positions[..., None, :, :] - positions[..., None, :, :, :]
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        contact_map = (dist < contact_threshold).float()
        contact_residue_idxs = torch.nonzero(contact_map, as_tuple=True)[0]

                

        return contact_residue_idxs


    def process_pdb(self, pdb_path=None,
                    chain_condition=None,
                    res_condition=None,
                    interface_cond=False,
                    contact_cond=False,                     
                    contact_threshold=8,
                    invert_contact_cond=False,
                ):
        """
        Processes a PDB file and generates input data for the model.

        Args:
            pdb_path (str, optional): Path to the PDB file.
            chain_condition (int or None, optional): If provided, select residues from this chain.
            res_condition (list of int or None, optional): If provided, select these residue indices.

        Returns:
            tuple: Processed data and sequence prior ready for model input.
        """
        if pdb_path is None:
            raise ValueError("pdb_path must be provided")
        data = self.data_pipe.process_pdb_multimer(pdb_path)
        data["mask_everything"] = np.array(0.).astype(np.float32)

        ground_truth = self.feature_pipeline_func.process_features(data, "predict")

        asym_id = ground_truth["asym_id"].squeeze(-1)
        residue_index = ground_truth["residue_index"].squeeze(-1)
        

        # Create seq_prior here
        seq_prior = torch.zeros_like(ground_truth["target_feat"][..., -1])
        seq_prior[..., -1] = 1  # Set the last dimension to 1

        # Handle chain_condition and res_condition
        if chain_condition is None and res_condition is None:
            mask = torch.zeros_like(residue_index, dtype=torch.bool)
        else:
            if chain_condition is None:
                chain_mask = torch.zeros_like(asym_id, dtype=torch.bool)
            else:
                def letter_to_num(char):
                    "convert letter to int"
                    if char.isalpha() and len(char) == 1:
                        # Convert to uppercase to handle both cases uniformly
                        return ord(char.upper()) - ord('A')
                    else:
                        raise ValueError("Input must be a single alphabet letter.")
                
                chain_condition_num = letter_to_num(chain_condition)
                chain_mask = (asym_id == chain_condition_num)
            if res_condition is None:
                res_mask = torch.zeros_like(residue_index, dtype=torch.bool)
            else:                
                res_mask = torch.isin(residue_index, torch.tensor(res_condition) -1) #shift by 1 because residue_index is 0 indexed

            mask = chain_mask | res_mask
        
        if interface_cond:
            interface_residues = get_interface_residues(ground_truth["all_atom_positions"][...,-1], ground_truth["all_atom_mask"][...,-1], ground_truth["asym_id"][...,-1], interface_threshold=contact_threshold)
            interface_mask = torch.isin(residue_index, interface_residues)            
            if invert_contact_cond:
                interface_mask = ~interface_mask
            mask = mask | interface_mask

        if contact_cond:
            contacts = (torch.mean(ground_truth["avg_distances"][...,-1], dim=-1) < contact_threshold).to(torch.int32)
            contact_mask = torch.isin(residue_index, contacts)            
            if invert_contact_cond:
                contact_mask = ~contact_mask
            mask = mask | contact_mask
            
        # Apply mask to seq_prior                
        seq_prior[mask] = ground_truth["target_feat"][..., -1][mask]

        #update diffuse mask        
        # data["diffuse_mask"][mask] = 0 
        ground_truth["diffuse_mask"][mask] = 0 
                
        # Add batch dimension
        data = tensor_tree_map(lambda x: x[None], ground_truth)
        seq_prior = seq_prior[None]  # Add batch dimension to seq_prior
        seq_prior = torch.argmax(seq_prior, dim = -1)
                        
        return data, seq_prior

    def calculate_recovery(self, true_seq, pred_seq):

        count = 0
        assert len(true_seq) == len(pred_seq), "Sequences must have the same length"

        seq_length = len(true_seq)
        for i in range(len(true_seq)):
            if (true_seq[i] == pred_seq[i]):
                count += 1

        seq_recovery = str(round(float(count / (seq_length)), 4))

        return seq_recovery, seq_length

    def predict(
        self,
        data=None,
        seq_prior=None,
        packed_path=None,
        seqs_path=None,
        noise_std=None,
        add_rigid_noise=False,
        prediction_index=1,
        pdb_path=None,  # Needed for output paths
        temp=0.1,      
        exclude_colon=False,
        omit_AA=None,
        tied_weights=False,
        sample_priority=False,
        run_cfg=False,
        sample_purity=False,
        partial_flows=False,
        t=None
    ):
        """
        Runs the model on the provided data and returns the predicted sequence.

        Args:
            data (dict): Input data for the model.
            seq_prior (torch.Tensor): The sequence prior.
            packed_path (str): Path to write PDB files.
            seqs_path (str): Path to write FASTA files.
            noise_std (float): Standard deviation of the noise to add.
            add_rigid_noise (bool): Whether to add rigid noise.
            prediction_index (int): Index of the current prediction for file naming.
            pdb_path (str): Path to the PDB file, used for output file naming.

        Returns:
            str: The predicted amino acid sequence.
        """
        if data is None or seq_prior is None:
            raise ValueError("data and seq_prior must be provided")

        # Clone data to avoid modifying the original
        data = tensor_tree_map(lambda x: x.clone(), data)
        # data = tensor_tree_map(lambda x: x.to(torch.bfloat16), data)
        seq_prior = seq_prior.clone()

        # Move data to device
        data = tensor_tree_map(lambda x: x.to(self.device), data)
        
        seq_prior = seq_prior.to(self.device)

        # If noise_std is provided, add noise to data["all_atom_positions"]
        if noise_std is not None and noise_std > 0:
            z = torch.randn_like(data["all_atom_positions"])
            data["all_atom_positions"] = data["all_atom_positions"] + (z * noise_std)

        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.no_grad():
            # Prepare inputs
            
            dtype = next(self.model.parameters()).dtype
            rigid_frames = (
                Rigid.from_tensor_4x4(data["backbone_rigid_tensor"][..., -1])
                .to_tensor_7()
                .to(self.device)
            )
            rigid_frames = scale_trans(rigid_frames, 0.1)
                        
            data["noise_label"] = torch.tensor([1. if noise_std else 0.], device=self.device)[None]            

            if omit_AA is not None:
                omit_AA_idx = [restype_order_with_x[aa] for aa in omit_AA]
            else:
                omit_AA_idx = None
                    
                    
            if partial_flows:
                prot_traj, conf = self.interpolant.partial_flows(data, self.model, rigid_frames, aa_init=seq_prior, temp=temp, omit_AA=omit_AA_idx, tied_weights=tied_weights, sample_priority=sample_priority, run_cfg=run_cfg, sample_purity=sample_purity, t=t)    
                unmasked_probs = None
            else:
                prot_traj, conf, unmasked_probs = self.interpolant.aa_sample(data, self.model, rigid_frames, aa_init=seq_prior, temp=temp, omit_AA=omit_AA_idx, tied_weights=tied_weights, sample_priority=sample_priority, run_cfg=run_cfg, sample_purity=sample_purity)
            predicted_seq = _aatype_to_str_sequence(prot_traj[-1][0])
            
            # Save unmasked_probs to JSON file in a separate json subdirectory
            if unmasked_probs is not None:
                output_prefix = os.path.splitext(os.path.basename(pdb_path))[0]
                json_path = os.path.join(os.path.dirname(seqs_path), "json")
                os.makedirs(json_path, exist_ok=True)
                probs_output_path = os.path.join(json_path, f"{output_prefix}_{prediction_index}_unmasked_probs.json")
                
                # Create pretty formatted JSON with amino acid names
                aa_names = list(restype_order_with_x.keys())
                probs_np = unmasked_probs.cpu().numpy()[0]  # Remove batch dimension, shape: (n_res, 21)
                
                # Build output dictionary with metadata
                output_data = {
                    "metadata": {
                        "pdb_path": pdb_path,
                        "ckpt_path": self.ckpt_path,
                        "temperature": temp,
                        "noise_std": noise_std,
                        "prediction_index": prediction_index,
                        "num_residues": probs_np.shape[0],
                        "run_cfg": run_cfg,
                        "sample_purity": sample_purity,
                        "sample_priority": sample_priority,
                        "tied_weights": tied_weights,
                        "omit_AA": omit_AA,
                    },
                    "residue_probabilities": {}
                }
                
                for res_idx in range(probs_np.shape[0]):
                    res_probs = probs_np[res_idx]
                    # Only include residues that have non-zero probabilities (were unmasked)
                    if res_probs.sum() > 0:
                        output_data["residue_probabilities"][f"residue_{res_idx + 1}"] = {
                            aa: round(float(prob), 6) for aa, prob in zip(aa_names, res_probs)
                        }
                
                with open(probs_output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
            
            #create a new mask to only have the backbone atoms
            atom_mask = torch.zeros_like(data["all_atom_mask"])
            atom_mask[:,:,[0,1,2,4],:] = 1
            data["all_atom_mask"] = data["all_atom_mask"] * atom_mask

            # Prepare the predicted protein structure
            predicted_protein = protein.Protein(
                atom_positions=data["noiseless_all_atom_positions"].cpu().numpy()[0, ..., 0],
                atom_mask=data["all_atom_mask"].cpu().numpy()[0, ..., 0],                
                aatype=prot_traj[-1][0].cpu().numpy(),
                residue_index=data["residue_index"].cpu().numpy()[0, ..., 0], 
                chain_index=data["asym_id"].cpu().numpy()[0, ..., 0],
                b_factors=torch.ones(
                    data["all_atom_positions"].shape[1], data["all_atom_positions"].shape[2]
                ),
            )

            pdb_string = protein.to_pdb(predicted_protein)
            true_seq = _aatype_to_str_sequence(torch.argmax(data["target_feat"][..., -1], dim=-1)[0])
            seq_recovery, seq_len = self.calculate_recovery(true_seq, predicted_seq)

            # Determine output paths
            output_prefix = os.path.splitext(os.path.basename(pdb_path))[0]
            output_suffix = f"_{prediction_index}"
            unrelaxed_path = os.path.join(packed_path, f"{output_prefix}{output_suffix}.pdb")
            fa_output_file = os.path.join(seqs_path, f"{output_prefix}{output_suffix}.fa")

            # Write PDB file
            with open(unrelaxed_path, 'w') as fp:
                fp.write(pdb_string)

            if not exclude_colon:
                col_index = torch.diff(data["asym_id"][...,-1])
                col_index = (col_index != 0).nonzero(as_tuple=True)[1]
                
                if col_index.numel() != 0:
                    insertion_points = (col_index + 1).tolist()
                    for idx in sorted(insertion_points, reverse=True):
                        predicted_seq = predicted_seq[:idx] + ":" + predicted_seq[idx:]
            

            # Determine sequence_id
            sequence_id = f"{output_prefix}{output_suffix}"            
            sequence = f">{sequence_id}, noise_std:{noise_std}, temp:{temp}, seq_recovery:{seq_recovery}, conf:{conf}, seq_len:{seq_len}\n{''.join(predicted_seq)}\n"

            return sequence


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Trifold Predictor Script")
    parser.add_argument(
        '--pdb_path',
        type=str,
        default=None,
        help='Path to the input PDB file.'
    )
    parser.add_argument(
        '--pdb_dir',
        type=str,
        default=None,
        help='Path to directory containing PDB files to process. If provided, all PDB files in this directory will be processed.'
    )
    parser.add_argument(
        '--json_path',  # New argument for JSON file
        type=str,
        default=None,
        help='Path to the JSON file containing PDB paths.'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default="./weights/afdb_dataset/afdb_weights.pt",
        help='Path to the model checkpoint file. Defaults to the example path if not provided.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help="Device to run the model on (e.g., 'cpu' or 'cuda:0')."
    )
    parser.add_argument(
        '--chain_condition',
        type=str,
        default=None,
        help='Chain condition to select specific chain. If not provided, all chains are selected.'
    )
    parser.add_argument(
        '--res_condition',
        nargs='*',
        type=int,
        default=None,
        help='Residue indices to select. If not provided, all residues are selected.'
    )
    parser.add_argument(
        '--output_root_dir',
        type=str,
        default="./",
        help='Root directory for output files. If not provided, outputs will be saved in "./redesigned/".'
    )
    parser.add_argument(
        '--num_predictions',
        type=int,
        default=8,
        help='Number of predictions to run per input file.'
    )

    parser.add_argument(
        '--noise_std',
        type=float,
        default=0,
        help='Noise standard deviation to add to all atoms.'
    )

    parser.add_argument(
        '--add_rigid_noise',
        type=bool,
        default=False,
        help='Add rigid noise.'
    )

    # New argument for half_half option
    parser.add_argument(
        '--half_half',
        action='store_true',
        help='Run half of the predictions with noise_std=0 and the other half with noise_std=0.2'
    )

    parser.add_argument(
        '--temp',
        type=float,
        default = 0.1,
        help='Control temperature parameter during sampling'
    )

    parser.add_argument(
        '--exclude_colon',
        action='store_true',
        help='Exclude ":" from the predicted sequences'
    )
    
    parser.add_argument(
        '--omit_AA',
        nargs='*',
        type=str,
        default=None,
        help='List of one-letter codes for amino acids to omit during sampling (e.g., C for Cysteine).'
    )    

    parser.add_argument(
        '--tied_weights',
        action='store_true',
        help='Use tied weights for the predictions.'
    )

    parser.add_argument(
        '--sample_priority',
        action='store_true',
        help='Sample priority for the predictions.'
    )
    parser.add_argument(
        '--cfg',
        action='store_true',
        help='Run cfg for the predictions.'
    )
    parser.add_argument(
        '--sample_purity',
        action='store_true',
        help='Sample purity for the predictions.'
    )
    parser.add_argument(
        '--partial_flows',
        action='store_true',
        help='Run partial flows for the predictions.'
    )
    parser.add_argument(
        '--t',
        type=float,
        default=0.5,
        help='forward diffusion time for the predictions.'
    )
    parser.add_argument(
        '--interface_cond',
        action='store_true',
        help='Redesign the interface for the predictions.'
    )
    parser.add_argument(
        '--contact_cond',
        action='store_true',
        help='Redesign the contact for the predictions.'
    )
    parser.add_argument(
        '--contact_threshold',
        type=float,
        default=8.0,
        help='Contact threshold for the predictions.'
    )
    parser.add_argument(
        '--invert_contact_cond',
        action='store_true',
        help='Invert the contact condition for the predictions.'
    )

    

    args = parser.parse_args()

    packed_path = os.path.join(args.output_root_dir, "backbones")  # Path to write PDB files
    seqs_path = os.path.join(args.output_root_dir, "seqs")  # Path to write FASTA files
    os.makedirs(packed_path, exist_ok=True)
    os.makedirs(seqs_path, exist_ok=True)

    predictor = TriFoldPredictor(
        ckpt_path=args.ckpt_path,
        device=args.device
    )
    
    # Determine noise_std list based on half_half option
    if args.half_half:
        half = args.num_predictions // 2
        noise_stds = [0] * half + [0.2] * (args.num_predictions - half)
    else:
        noise_stds = [args.noise_std] * args.num_predictions

    if args.pdb_dir:
        # Process all PDB files in the directory
        pdb_files = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
        # Use tqdm to display a progress bar over PDB files

        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            try:
                data, seq_prior = predictor.process_pdb(
                    pdb_path=pdb_file,
                    chain_condition=args.chain_condition,
                    res_condition=args.res_condition,
                    interface_cond=args.interface_cond,
                    contact_cond=args.contact_cond,
                    contact_threshold=args.contact_threshold,
                    invert_contact_cond=args.invert_contact_cond
                )
                output_prefix = os.path.splitext(os.path.basename(pdb_file))[0]
                output_path = os.path.join(seqs_path, f"{output_prefix}.fa")
                true_seq = _aatype_to_str_sequence(torch.argmax(data["target_feat"][..., -1], dim=-1)[0]) 
                if not args.exclude_colon:
                    #figure out where to add ":"
                    col_index = torch.diff(data["asym_id"][...,-1])
                    col_index = (col_index != 0).nonzero(as_tuple=True)[1]
                    
                    if col_index.numel() != 0:
                        insertion_points = (col_index + 1).tolist()
                        for idx in sorted(insertion_points, reverse=True):
                            true_seq = true_seq[:idx] + ":" + true_seq[idx:]

                header = f">{output_prefix}, ckpt_path:{args.ckpt_path}, seq_len:{len(true_seq)}\n{''.join(true_seq)}\n"
                all_predictions = [header]

                for i in tqdm(range(args.num_predictions), desc="Predictions"):
                    try:
                        current_noise_std = noise_stds[i] if args.half_half else args.noise_std
                        predicted_sequence = predictor.predict(
                            data=data,
                            seq_prior=seq_prior,
                            packed_path=packed_path,
                            seqs_path=seqs_path,
                            prediction_index=i,
                            noise_std=current_noise_std,
                            pdb_path=pdb_file,  # For output paths, 
                            temp = args.temp,
                            exclude_colon=args.exclude_colon,
                            omit_AA=args.omit_AA,
                            tied_weights=args.tied_weights,
                            sample_priority=args.sample_priority,
                            run_cfg=args.cfg,
                            sample_purity=args.sample_purity,
                            partial_flows=args.partial_flows,
                            t=args.t
                        )                
                        all_predictions.append(predicted_sequence)
                    except Exception as e:
                        print(f"Error during prediction {i} for {pdb_file}: {str(e)}")
                        continue

                # Write FASTA file here
                try:
                    with open(output_path, "w") as fasta_file:
                        fasta_file.writelines(all_predictions)
                except Exception as e:
                    print(f"Error writing FASTA file for {pdb_file}: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")
                continue

    elif args.pdb_path:
        # Process a single PDB file
        data, seq_prior = predictor.process_pdb(
            pdb_path=args.pdb_path,
            chain_condition=args.chain_condition,
            res_condition=args.res_condition,
            interface_cond=args.interface_cond,
            contact_cond=args.contact_cond,
            contact_threshold=args.contact_threshold,
            invert_contact_cond=args.invert_contact_cond
        )

        
        output_prefix = os.path.splitext(os.path.basename(args.pdb_path))[0]
        output_path = os.path.join(seqs_path, f"{output_prefix}.fa")
        true_seq = _aatype_to_str_sequence(torch.argmax(data["target_feat"][..., -1], dim=-1)[0]) 
        #figure out where to add ":"
        if not args.exclude_colon:        
            col_index = torch.diff(data["asym_id"][...,-1])
            col_index = (col_index != 0).nonzero(as_tuple=True)[1]
            
            if col_index.numel() != 0:
                insertion_points = (col_index + 1).tolist()
                for idx in sorted(insertion_points, reverse=True):
                    true_seq = true_seq[:idx] + ":" + true_seq[idx:]


        
        header = f">{output_prefix}, ckpt_path:{args.ckpt_path}, seq_len:{len(true_seq)}\n{''.join(true_seq)}\n"
        
        start_time = time.time()
        all_predictions = [header]
        for i in tqdm(range(args.num_predictions), desc="Predictions"):
            current_noise_std = noise_stds[i] if args.half_half else args.noise_std
            predicted_sequence = predictor.predict(
                data=data,
                seq_prior=seq_prior,
                packed_path=packed_path,
                seqs_path=seqs_path,
                prediction_index=i,
                noise_std=current_noise_std,
                pdb_path=args.pdb_path,  # For output paths
                temp = args.temp, 
                exclude_colon=args.exclude_colon,
                omit_AA=args.omit_AA,
                tied_weights=args.tied_weights, 
                sample_priority=args.sample_priority,
                run_cfg=args.cfg,
                sample_purity=args.sample_purity,
                partial_flows=args.partial_flows,
                t=args.t,                
            )

            print(f"\nPredicted Sequence {i}:")
            print(predicted_sequence)
            all_predictions.append(predicted_sequence)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"time for {args.num_predictions} took {elapsed_time:.2f} seconds")

        with open(output_path, "w") as fasta_file:
            fasta_file.writelines(all_predictions)

    elif args.json_path:
        # Process all PDB files listed in the JSON file
        with open(args.json_path, 'r') as f:
            pdb_dict = json.load(f)
        pdb_files = list(pdb_dict.keys())

        for pdb_file in tqdm(pdb_files, desc="Processing PDB files from JSON"):
            try:
                data, seq_prior = predictor.process_pdb(
                    pdb_path=pdb_file,
                    chain_condition=args.chain_condition,
                    res_condition=args.res_condition,
                    interface_cond=args.interface_cond,
                    contact_cond=args.contact_cond,
                    contact_threshold=args.contact_threshold,
                    invert_contact_cond=args.invert_contact_cond
                )
                output_prefix = os.path.splitext(os.path.basename(pdb_file))[0]
                output_path = os.path.join(seqs_path, f"{output_prefix}.fa")
                
                true_seq = _aatype_to_str_sequence(torch.argmax(data["target_feat"][..., -1], dim=-1)[0]) 

                #figure out where to add ":"
                if not args.exclude_colon:                    
                    col_index = torch.diff(data["asym_id"][...,-1])
                    col_index = (col_index != 0).nonzero(as_tuple=True)[1]
                    
                    if col_index.numel() != 0:
                        insertion_points = (col_index + 1).tolist()
                        for idx in sorted(insertion_points, reverse=True):
                            true_seq = true_seq[:idx] + ":" + true_seq[idx:]

                header = f">{output_prefix}, ckpt_path:{args.ckpt_path}, seq_len:{len(true_seq)}\n{''.join(true_seq)}\n"
                all_predictions = [header]

                for i in tqdm(range(args.num_predictions), desc="Predictions"):
                    try:
                        current_noise_std = noise_stds[i] if args.half_half else args.noise_std
                        predicted_sequence = predictor.predict(
                            data=data,
                            seq_prior=seq_prior,
                            packed_path=packed_path,
                            seqs_path=seqs_path,
                            prediction_index=i,
                            noise_std=current_noise_std,
                            pdb_path=pdb_file,   # For output paths
                            temp = args.temp,                     
                            exclude_colon=args.exclude_colon,
                            omit_AA=args.omit_AA,
                            tied_weights=args.tied_weights, 
                            sample_priority=args.sample_priority,
                            run_cfg=args.cfg,
                            sample_purity=args.sample_purity,
                            partial_flows=args.partial_flows,
                            t=args.t
                        )
                        all_predictions.append(predicted_sequence)
                    except Exception as e:
                        print(f"Error during prediction {i} for {pdb_file}: {str(e)}")
                        continue

                # Write FASTA file here
                try:
                    with open(output_path, "w") as fasta_file:
                        fasta_file.writelines(all_predictions)
                except Exception as e:
                    print(f"Error writing FASTA file for {pdb_file}: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")
                continue

    else:
        print("Error: Please provide either --pdb_path, --pdb_dir, or --json_path")
        return


if __name__ == "__main__":
    main()
