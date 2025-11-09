import dataclasses
import numpy as np
import collections
import string
import pickle
import os
import torch
from typing import List, Dict, Any
from triflow.utils import rigid_utils as ru
from Bio.PDB.Chain import Chain
from Bio import PDB
from triflow.np import protein, residue_constants
from glob import glob


Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

NUM_TOKENS = residue_constants.restype_num
MASK_TOKEN_INDEX = residue_constants.restypes_with_x.index('X')
CA_IDX = residue_constants.atom_order['CA']

