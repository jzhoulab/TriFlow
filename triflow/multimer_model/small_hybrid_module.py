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
import math
import sys
import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional
from functools import partial

from triflow.multimer_model.primitives import Linear, LayerNorm
from triflow.multimer_model.dropout import DropoutRowwise, DropoutColumnwise
from triflow.multimer_model.msa import (
    MSARowAttentionWithPairBias,
)
# from triflow.multimer_model.outer_product_mean import OuterProductMean
from triflow.multimer_model.pair_transition import PairTransition

from triflow.multimer_model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from triflow.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
from triflow.utils.chunk_utils import chunk_layer
from triflow.utils.tensor_utils import add

from triflow.multimer_model.hybrid_structure_module import (
    InvariantPointAttention, 
    BackboneUpdate,
    StructureModuleTransition,
)

from triflow.utils.rigid_utils import Rigid



class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(self, c_s, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_s = c_s
        self.n = n

        self.layer_norm = LayerNorm(self.c_s)
        self.linear_1 = Linear(self.c_s, self.n * self.c_s, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_s, self.c_s, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
         )


    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class OuterSum(nn.Module):
    def __init__(self, c_s, c_z):
        super(OuterSum, self).__init__()
        self.linear_i = nn.Linear(c_s, c_z)
        self.linear_j = nn.Linear(c_s, c_z)
        
    def forward(self,m):
        p_i = self.linear_i(m)
        p_j = self.linear_j(m)

        # [b, n_res, n_res, c_p]
        p = p_i[:, :, None, :] + p_j[:, None, :, :]

        return p


class HybridBlock(nn.Module):
    def __init__(self,
        c_z: int,
        c_hidden_single_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_single: int,
        no_heads_pair: int,
        transition_n: int,
        single_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        c_s,
        c_ipa, 
        no_heads_ipa, 
        no_qk_points, 
        no_v_points,
        no_transition_layers, 
        dropout_rate, 
        **kwargs,
    ):
        super(HybridBlock, self).__init__()

        self.single_att_row = MSARowAttentionWithPairBias(
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden_single_att,
            no_heads=no_heads_single,
            inf=inf,
        )

        self.single_transition = MSATransition(
            c_s=c_s,
            n=transition_n,
        )
        
        self.outer_sum = OuterSum(
        c_s = c_s,
        c_z = c_z,
        )

        self.ipa = InvariantPointAttention(
            c_s,
            c_z,
            c_ipa,
            no_heads_ipa,
            no_qk_points,
            no_v_points,
            inf=inf,
            eps=eps,
        )
       
        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa = LayerNorm(c_s)

        self.transition = StructureModuleTransition(
            c_s,
            no_transition_layers,
            dropout_rate,
        )

        self.bb_update = BackboneUpdate(c_s)        

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )
        

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

        self.linear_pb = Linear(3, c_z)



    def forward(self,
        s: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        rigid_frames,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        if(_offload_inference and inplace_safe):
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [s, z]

        s, z = input_tensors
        
        #run single track
        s = add(s,
            self.single_att_row(
                s,
                z = z,
                mask=seq_mask,
                chunk_size=chunk_size,
                use_lma=use_lma,
                use_flash=use_flash,
            ),
            inplace=inplace_safe,
        )
               
        if(not inplace_safe):
            input_tensors = [s, input_tensors[1]]
        

        s, z = input_tensors
        s = add(
            s,
            self.single_transition(
                s, #mask=msa_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )         

        outer_sum = self.outer_sum(
            s, 
        )

        # share from single to pair
        z = add(z, outer_sum, inplace=inplace_safe)
        del outer_sum        

        #run ipa
        rigid_frames = Rigid.from_tensor_7(rigid_frames)
        s = s + self.ipa(
           s,
           z,
           rigid_frames,
           seq_mask,
           inplace_safe=inplace_safe,
           _offload_inference=_offload_inference,
           _z_reference_list= None
        )

        s = self.ipa_dropout(s)
        s = self.layer_norm_ipa(s)
        s = self.transition(s)

        rigid_frames = rigid_frames.compose_q_update_vec(self.bb_update(s))        
        

        pair_bias = rigid_frames.invert()[..., None].apply(
            rigid_frames.get_trans()[..., None, :, :],
        )
        
        pair_bias = self.linear_pb(pair_bias)
        z = add(z, pair_bias, inplace=inplace_safe)        
        
        #run pair track
        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if(not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update
        
        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if(not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update
       
        del tmu_update

        #convert rigids to res x res for bias in triangle attention





        if(inplace_safe):
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]

        z = add(z,
            self.pair_transition(
                z, #mask=pair_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        return s, z, rigid_frames.to_tensor_7()

class HybridStack(nn.Module):
    def __init__(self,
        c_z: int,
        c_hidden_single_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_single: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        single_dropout: float,
        pair_dropout: float,
        blocks_per_ckpt,
        inf: float,
        eps: float,
        c_s,
        c_ipa,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        no_transition_layers,
        dropout_rate,
        **kwargs,
    ):
        super(HybridStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt        
        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = HybridBlock(
                c_z=c_z,
                c_hidden_single_att=c_hidden_single_att,
                c_hidden_opm=c_hidden_opm,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_single=no_heads_single,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                single_dropout=single_dropout,
                pair_dropout=pair_dropout,
                inf=inf,
                eps=eps,
                c_s=c_s,
                c_ipa=c_ipa,
                no_heads_ipa=no_heads_ipa,
                no_qk_points=no_qk_points,
                no_v_points=no_v_points,
                no_transition_layers=no_transition_layers,
                dropout_rate=dropout_rate,
            )
            self.blocks.append(block)        


    def _prep_blocks(self, 
        s: torch.Tensor, 
        z: torch.Tensor, 
        rigid_frames, 
        chunk_size: int,
        use_lma: bool,
        use_flash: bool,
        seq_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b,
                seq_mask=seq_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_lma=use_lma,
                use_flash=use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]
        return blocks

    def forward(self,
        s: torch.Tensor,
        z: torch.Tensor,
        rigid_frames,
        seq_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_lma: Whether to use low-memory attention during inference
            use_flash: 
                Whether to use FlashAttention where possible. Mutually 
                exclusive with use_lma.
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """ 

        blocks = self._prep_blocks(
            s=s,
            z=z,
            rigid_frames=rigid_frames,
            chunk_size=chunk_size,
            use_lma=use_lma,
            use_flash=use_flash,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        # blocks_per_ckpt = self.blocks_per_ckpt
#        if(not torch.is_grad_enabled()):
        blocks_per_ckpt = None
        s, z, rigid_frames = checkpoint_blocks(
            blocks,
            args=(s, z, rigid_frames),
            blocks_per_ckpt=blocks_per_ckpt,
        )
            

        return s, z, rigid_frames 

