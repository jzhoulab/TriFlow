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

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from einops import repeat, rearrange

from triflow.multimer_model.primitives import Linear, LayerNorm
from triflow.utils.tensor_utils import add, one_hot

from triflow.multimer_model.dropout import DropoutRowwise, DropoutColumnwise
from triflow.multimer_model.triangular_attention import (
    TriangleAttention,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from triflow.multimer_model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)

from torch.utils.checkpoint import checkpoint


    

#updated with checkpoint
class TriangleBlock(nn.Module):
    def __init__(self,
        c_z, 
        c_hidden_mul, 
        no_heads_pair, 
        inf, 
        pair_dropout, 
        **kwargs,
    ):
        super(TriangleBlock, self).__init__()

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttention(
            c_z,
            32,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            32,
            no_heads_pair,
            inf=inf,
        )          

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

    def forward(self, z, pair_mask, inplace_safe=False, _attn_chunk_size=None, use_lma=False, use_checkpointing=True):
        # Define checkpointed functions
        def tmu_out_fn(z, pair_mask):
            tmu_update = self.tri_mul_out(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
            )
            if not inplace_safe:
                z_new = z + self.ps_dropout_row_layer(tmu_update)
            else:
                z_new = tmu_update
            return z_new

        def tmu_in_fn(z, pair_mask):
            tmu_update = self.tri_mul_in(
                z,
                mask=pair_mask,
                inplace_safe=inplace_safe,
                _add_with_inplace=True,
            )
            if not inplace_safe:
                z_new = z + self.ps_dropout_row_layer(tmu_update)
            else:
                z_new = tmu_update
            return z_new

        def tri_att_start_fn(z, pair_mask):
            attn_out = self.ps_dropout_row_layer(
                self.tri_att_start(
                    z, 
                    mask=pair_mask,
                    chunk_size=_attn_chunk_size, 
                    use_memory_efficient_kernel=False, 
                    use_lma=use_lma, 
                    inplace_safe=inplace_safe
                )
            )
            z_new = z + attn_out if not inplace_safe else add(z, attn_out, inplace=True)
            return z_new

        def tri_att_end_fn(z, pair_mask_t):
            attn_out = self.ps_dropout_col_layer(
                self.tri_att_end(
                    z,
                    mask=pair_mask_t,
                    chunk_size=_attn_chunk_size, 
                    use_memory_efficient_kernel=False, 
                    use_lma=use_lma, 
                    inplace_safe=inplace_safe
                )
            )
            z_new = z + attn_out if not inplace_safe else add(z, attn_out, inplace=True)
            return z_new

        # Apply checkpointing to each computational block if enabled
        if torch.is_grad_enabled() and use_checkpointing:
            # Apply checkpointing during training
            z = checkpoint(tmu_out_fn, z, pair_mask)
            z = checkpoint(tmu_in_fn, z, pair_mask)
            z = checkpoint(tri_att_start_fn, z, pair_mask)
        else:
            # Directly call the functions during validation/inference
            z = tmu_out_fn(z, pair_mask)
            z = tmu_in_fn(z, pair_mask)
            z = tri_att_start_fn(z, pair_mask)

        z = z.transpose(-2, -3)

        if torch.is_grad_enabled() and use_checkpointing:
            z = checkpoint(tri_att_end_fn, z, pair_mask.transpose(-1, -2))
        else:
            z = tri_att_end_fn(z, pair_mask.transpose(-1, -2))

        z = z.transpose(-2, -3)

        return z



    
class TimeEmbedding(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, num_channels, scale=30.):
        super(TimeEmbedding, self).__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.l1 = Linear(embed_dim, embed_dim)
        self.dense1 = Linear(embed_dim, num_channels)


    def forward(self, x):
        x_proj = x[:,None] * self.W[None,:] * 2 * np.pi
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).to(self.W.dtype)        
        return (self.dense1(x)) #[batch, channels]

class RelPosEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        c_z: int,
        relpos_k: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(RelPosEmbedder, self).__init__()


        self.c_z = c_z

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        ) 
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)
    


    
    def forward(
        self,
        ri: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """

        pair_emb = self.relpos(ri.float())
#        single_emb = self.linear_single(single)

        return pair_emb



class RelPosEmbedderMultimer(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        c_z: int,
        relpos_k: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(RelPosEmbedderMultimer, self).__init__()



        self.c_z = c_z

        # RPE multimer stuff

        use_chain_relative = True
        max_relative_idx = 32
        max_relative_chain= 2

        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if(self.use_chain_relative):

            self.no_bins = (
                2 * max_relative_idx + 2
            )            
        else:
            self.no_bins = 2 * max_relative_idx + 1
        

        #rbf stuff
        self.dist_bins = 24
        self.dist_bin_width = 0.5
        self.rbf_final_dim = 15 * self.dist_bins
        self.linear_relpos = Linear(self.rbf_final_dim + self.no_bins, c_z)        



    def rbf(self, D):
        """
        Radial basis functions.
        """
        
        # dist_bins = 24
        # dist_bin_width = 0.5
            
        device = D.device
        D_min, D_max, D_count = 0., self.dist_bins * self.dist_bin_width, self.dist_bins
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF


    def relpos(self, batch):

        use_chain_relative = True
        max_relative_idx = 32
        max_relative_chain= 2
        
        
        pos = batch["residue_index"]
        asym_id = batch["asym_id"]
        asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(
            offset + max_relative_idx, 0, 2 * max_relative_idx
        )

        rel_feats = []
        if(use_chain_relative):
            final_offset = torch.where(
                asym_id_same, 
                clipped_offset,
                (2 * max_relative_idx + 1) * 
                torch.ones_like(clipped_offset)
            )
            boundaries = torch.arange(
                start=0, end=2 * max_relative_idx + 2, device=final_offset.device
            )
            rel_pos = one_hot(
                final_offset,
                boundaries,
            )

            rel_feats.append(rel_pos)
            
        else:
            boundaries = torch.arange(
                start=0, end=2 * max_relative_idx + 1, device=clipped_offset.device
            )
            rel_pos = one_hot(
                clipped_offset, boundaries,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(
            self.linear_relpos.weight.dtype
        )

        return rel_feat

    
    def forward(
        self,
        batch,
        inplace_safe: bool = False,
        is_training=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        atom14 = torch.nan_to_num(batch["all_atom_positions"])

        if atom14.shape[2] == 14:

            n_pos = atom14[...,0,:]
            ca_pos = atom14[...,1,:]
            c_pos = atom14[...,2,:]
            o_pos = atom14[...,3,:]
            
        else:
            n_pos = atom14[...,0,:]
            ca_pos = atom14[...,1,:]
            c_pos = atom14[...,2,:]
            o_pos = atom14[...,4,:] 

        b = ca_pos - n_pos
        c = c_pos - ca_pos
        a = torch.cross(b, c, dim=-1)
        cb_pos = -0.58273431*a + 0.56802827*b - 0.54067466*c + ca_pos        

        batch_dim, num_res, _ = n_pos.shape
        device = n_pos.device
        num_distance_matrix = 15
        



        nn_pos = self.rbf(torch.cdist(n_pos, n_pos))
        caca_pos = self.rbf(torch.cdist(ca_pos, ca_pos))
        cc_pos = self.rbf(torch.cdist(c_pos, c_pos))
        oo_pos = self.rbf(torch.cdist(o_pos, o_pos))
        cbcb_pos = self.rbf(torch.cdist(cb_pos, cb_pos))

        nca_pos = self.rbf(torch.cdist(n_pos, ca_pos))
        nc_pos = self.rbf(torch.cdist(n_pos, c_pos))
        no_pos = self.rbf(torch.cdist(n_pos, o_pos))
        ncb_pos = self.rbf(torch.cdist(n_pos, cb_pos))

        cac_pos = self.rbf(torch.cdist(ca_pos, c_pos))
        cao_pos = self.rbf(torch.cdist(ca_pos, o_pos))
        cacb_pos = self.rbf(torch.cdist(ca_pos, cb_pos))

        co_pos = self.rbf(torch.cdist(c_pos, o_pos))
        ccb_pos = self.rbf(torch.cdist(c_pos, cb_pos))

        ocb_pos = self.rbf(torch.cdist(o_pos, cb_pos))


    
        dist_matrix = torch.cat([nn_pos, caca_pos, cc_pos, oo_pos, cbcb_pos,
                                  nca_pos, nc_pos, no_pos, ncb_pos,
                                  cac_pos, cao_pos,cacb_pos,
                                  co_pos, ccb_pos, 
                                  ocb_pos
                   ], dim=-1)
    

        dist_matrix = dist_matrix.float()        
        dist_matrix = dist_matrix.to(batch["all_atom_positions"].dtype)        
        
        relpos = self.relpos(batch)
        pair_emb = torch.cat([relpos, dist_matrix], dim= -1)        
        pair_emb = self.linear_relpos(pair_emb)


        return pair_emb




class SingleOutEmbedder(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden):
        super(SingleOutEmbedder, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden,)
        self.linear_2 = Linear(self.c_hidden, self.c_hidden,)
        self.linear_3 = Linear(self.c_hidden, self.no_bins,)

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s





