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
import logging
import ml_collections
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import random


from triflow.utils.rigid_utils import Rotation, Rigid


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def scale_trans(tensor_7, scale_factor):
    """ Scale translation by a scale factor
    Args:
        tensor_6: A tensor of shape [*, NUM_RES, 6]
        scale__factor: A float to scale trans

    Returns:
        scaled tensor_6
    """

    tensor_7 = tensor_7.clone().detach()
    tensor_7[...,-3:] = tensor_7[...,-3:] * scale_factor

    return tensor_7

def add_trans_noise(tensor_7):
    """ Add noise to translation vector
    Args:
        tensor_7: A tensor of shape [*, NUM_RES, 7]
        scale__factor: A float to scale trans

    Returns:
         tensor_7
    """

    tensor_7 = tensor_7.clone().detach()
    batch_size, sample_size, _ = tensor_7.shape
    noise = ((torch.rand( (batch_size, sample_size, 3), device = tensor_7.device) * 2) - 1 ) / 5
    tensor_7[...,-3:] = tensor_7[...,-3:] + noise

    return tensor_7


def dgram_from_positions_ca(
    pos1: torch.Tensor, 
    pos2: torch.Tensor, 
    min_bin: float = 2.3125, 
    max_bin: float = 21.6875, 
    no_bins: float = 64, 
    inf: float = 1e6,            
):
    
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=pos1.device,
    )        

    boundaries = boundaries ** 2

    dists = torch.sum(
        (pos1[..., None, :] - pos2[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)     

    true_one_hot = torch.nn.functional.one_hot(true_bins, no_bins)

    return true_one_hot



def true_bb_distogram(feats):



    def dgram_from_positions(
        pos1: torch.Tensor, 
        pos2: torch.Tensor, 
        min_bin: float = 2.3125, 
        max_bin: float = 21.6875, 
        no_bins: float = 64, 
        inf: float = 1e6,            
    ):
        
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            no_bins - 1,
            device=pos1.device,
        )        

        boundaries = boundaries ** 2

        dists = torch.sum(
            (pos1[..., None, :] - pos2[..., None, :, :]) ** 2,
            dim=-1,
            keepdims=True,
        )

        true_bins = torch.sum(dists > boundaries, dim=-1)     

        true_one_hot = torch.nn.functional.one_hot(true_bins, no_bins)

        return true_one_hot


    
    
    if feats["noiseless_all_atom_positions"].shape[2] == 14:
        n_pos = feats["noiseless_all_atom_positions"][...,-1][...,0,:] 
        ca_pos = feats["noiseless_all_atom_positions"][...,-1][...,1,:]
        c_pos = feats["noiseless_all_atom_positions"][...,-1][...,2,:]
        o_pos = feats["noiseless_all_atom_positions"][...,-1][...,3,:]

    else:
        # we assume it is atom37 otherwise
        n_pos = feats["noiseless_all_atom_positions"][...,-1][...,0,:] 
        ca_pos = feats["noiseless_all_atom_positions"][...,-1][...,1,:]
        c_pos = feats["noiseless_all_atom_positions"][...,-1][...,2,:]
        o_pos = feats["noiseless_all_atom_positions"][...,-1][...,4,:]



    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = torch.cross(b, c, dim=-1)
    cb_pos = -0.58273431*a + 0.56802827*b - 0.54067466*c + ca_pos   


    nn_pos = dgram_from_positions(n_pos, n_pos)
    caca_pos = dgram_from_positions(ca_pos, ca_pos)

    cc_pos = dgram_from_positions(c_pos, c_pos)
    oo_pos = dgram_from_positions(o_pos, o_pos)
    cbcb_pos = dgram_from_positions(cb_pos, cb_pos)

    nca_pos = dgram_from_positions(n_pos, ca_pos)
    nc_pos = dgram_from_positions(n_pos, c_pos)
    no_pos = dgram_from_positions(n_pos, o_pos)
    ncb_pos = dgram_from_positions(n_pos, cb_pos)

    cac_pos = dgram_from_positions(ca_pos, c_pos)
    cao_pos = dgram_from_positions(ca_pos, o_pos)
    cacb_pos = dgram_from_positions(ca_pos, cb_pos)

    co_pos = dgram_from_positions(c_pos, o_pos)
    ccb_pos = dgram_from_positions(c_pos, cb_pos)

    ocb_pos = dgram_from_positions(o_pos, cb_pos)         

    dist_matrix = torch.cat([nn_pos, caca_pos, cc_pos, oo_pos, cbcb_pos,
                            nca_pos, nc_pos, no_pos, ncb_pos,
                            cac_pos, cao_pos,cacb_pos,
                            co_pos, ccb_pos, 
                            ocb_pos
            ], dim=-1)
            
    dist_matrix = dist_matrix.float()

    return dist_matrix
    


def calculate_loss(batch, x,frames, model,eps = 1e-5,):
    """
        x: A tensor of shape [*, NUM_RES, 6]
        evoformer_output_dict: Dictionary containing single and pair representation
        aatype:
        score_model:
        marginal_prob_std:
        eps:
    """

    frames = add_trans_noise(frames)
    scaled_frames = scale_trans(frames, 0.1)

    #model prediction
        
    outputs = model(batch,scaled_frames)
    loss = softmax_cross_entropy(outputs["all_logits"], x)
    loss = torch.sum((loss * batch["seq_mask"][...,-1]), dim = -1 ) / (eps + torch.sum(batch["seq_mask"][...,-1], dim = -1))
    loss = torch.mean(loss)

    #calculate distogram loss

    true_bb_dgram = true_bb_distogram(batch)
    pair_mask = batch["seq_mask"][...,-1][..., None] * batch["seq_mask"][...,-1][..., None, :]

    dgram_loss = softmax_cross_entropy(outputs["dgram"], true_bb_dgram)
    dgram_loss = torch.sum((dgram_loss * pair_mask), dim = -1 ) / (eps + torch.sum(pair_mask, dim = (-1, -2)))[...,None]    
    dgram_loss = torch.sum(dgram_loss, dim = -1)

    #average over the batch dimension
    dgram_loss = torch.mean(dgram_loss)

    return loss, dgram_loss, outputs
    


def calculate_multimer_loss(batch, x,frames, model,eps = 1e-5, seq=None, seq_mask=None):
    """
        x: A tensor of shape [*, NUM_RES, 6]
        evoformer_output_dict: Dictionary containing single and pair representation
        aatype:
        score_model:
        marginal_prob_std:
        eps:
    """

    # frames = add_trans_noise(frames)
    scaled_frames = scale_trans(frames, 0.1)

    #model prediction
    
    #give partial information during training 
    outputs = model(batch,scaled_frames, seq)
    
    loss = softmax_cross_entropy(outputs["all_logits"], x)    

    loss = torch.sum((loss * seq_mask), dim = -1 ) / (eps + torch.sum(seq_mask, dim = -1)) #seq_mask is based on diffuse_mask
    loss = torch.sum((loss * (1 - batch["mask_everything"][...,-1])), dim = -1 ) / (eps + torch.sum((1 - batch["mask_everything"][...,-1]), dim = -1))
    loss = torch.mean(loss)


    #calculate distogram loss
    true_bb_dgram = true_bb_distogram(batch)
    pair_mask = batch["seq_mask"][...,-1][..., None] * batch["seq_mask"][...,-1][..., None, :]
    dgram_loss = softmax_cross_entropy(outputs["dgram"], true_bb_dgram)
    dgram_loss = torch.sum((dgram_loss * pair_mask), dim = -1 ) / (eps + torch.sum(pair_mask, dim = (-1, -2)))[...,None]    
    dgram_loss = torch.sum(dgram_loss, dim = -1)

    #average over the batch dimension
    dgram_loss = torch.mean(dgram_loss) * 0.01 * 0.1

    if torch.isnan(loss) or torch.isinf(loss):
        loss = loss.new_tensor(0., requires_grad=True)
        print("Loss is nan or inf")

    if torch.isnan(dgram_loss) or torch.isinf(dgram_loss):
        dgram_loss = dgram_loss.new_tensor(0., requires_grad=True)
        print("Dgram loss is nan or inf")
    print(loss)
    return loss, dgram_loss, outputs

    # return loss, outputs    



