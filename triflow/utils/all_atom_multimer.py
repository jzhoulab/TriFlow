from functools import partial
from typing import Dict, Text, Tuple

import torch

from triflow.np import multimer_residue_constants as rc
from triflow.utils import tensor_utils
import numpy as np


def squared_difference(x, y):
    return np.square(x - y)


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]


def atom14_to_atom37(
    atom14_data: torch.Tensor,  # (*, N, 14, ...)
    aatype: torch.Tensor # (*, N)
) -> Tuple:    # (*, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    idx_atom37_to_atom14 = get_rc_tensor(rc.RESTYPE_ATOM37_TO_ATOM14, aatype).long()
    no_batch_dims = len(aatype.shape) - 1
    atom37_data = tensor_utils.batched_gather(
        atom14_data, 
        idx_atom37_to_atom14, 
        dim=no_batch_dims + 1, 
        no_batch_dims=no_batch_dims + 1
    )
    atom37_mask = get_rc_tensor(rc.RESTYPE_ATOM37_MASK, aatype) 
    if len(atom14_data.shape) == no_batch_dims + 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == no_batch_dims + 3:
        atom37_data *= atom37_mask[..., None].to(dtype=atom37_data.dtype)
    else:
        raise ValueError("Incorrectly shaped data")
    return atom37_data, atom37_mask


def atom37_to_atom14(aatype, all_atom_pos, all_atom_mask):
    """Convert Atom37 positions to Atom14 positions."""
    residx_atom14_to_atom37 = get_rc_tensor(
        rc.RESTYPE_ATOM14_TO_ATOM37, aatype
    )
    no_batch_dims = len(aatype.shape)
    atom14_mask = tensor_utils.batched_gather(
        all_atom_mask, 
        residx_atom14_to_atom37, 
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    ).to(all_atom_pos.dtype)
    # create a mask for known groundtruth positions
    atom14_mask *= get_rc_tensor(rc.RESTYPE_ATOM14_MASK, aatype) 
    # gather the groundtruth positions
    atom14_positions = tensor_utils.batched_gather(
        all_atom_pos, 
        residx_atom14_to_atom37, 
        dim=no_batch_dims + 1,
        no_batch_dims=no_batch_dims + 1,
    ),
    atom14_positions = atom14_mask * atom14_positions
    return atom14_positions, atom14_mask
