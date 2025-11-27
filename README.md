![Header](imgs/header.png)

A model for protein sequence prediction given backbone coordinates 

## Installation (Linux)

All Python dependencies are specified in `environment.yml`.  

```bash
mamba env create -f environment.yaml
```

## Inference

### Basic Usage

```bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
  --pdb_path <path_to_pdb> \
  --ckpt_path ./weights/mpnn_dataset/epoch\=66-step\=167499.pt \
  --num_predictions 8 \
  --output_root_dir ./scratch/
```

### Command Line Options

#### Input Options
- `--pdb_path`: Path to a single input PDB file
- `--pdb_dir`: Path to directory containing multiple PDB files to process
- `--json_path`: Path to JSON file containing list of PDB file paths
- `--ckpt_path`: Path to model checkpoint file (default: `./weights/afdb_dataset/epoch=90-step=90999.pt`)
- `--device`: Device to run on, e.g., 'cuda:0' or 'cpu' (default: `cuda:0`)

#### Output Options
- `--output_root_dir`: Root directory for output files (default: `./`)
  - Creates subdirectories: `backbones/` (PDB files) and `seqs/` (FASTA files)
- `--exclude_colon`: Exclude ":" separator between chains in output sequences

#### Sampling Options
- `--num_predictions`: Number of predictions per input structure (default: `8`)
- `--temp`: Temperature parameter for sampling (default: `0.1`)
- `--noise_std`: Standard deviation of Gaussian noise to add to coordinates (default: `0.0`)
- `--half_half`: Run half predictions with noise_std=0 and half with noise_std=0.2

#### Conditioning Options
- `--chain_condition`: Condition on specific chain (e.g., 'A', 'B')
- `--res_condition`: Space-separated list of residue indices to condition on (e.g., `--res_condition 1 2 3 10 15`)

#### Amino Acid Constraints
- `--omit_AA`: Space-separated list of amino acid one-letter codes to exclude from sampling (e.g., `--omit_AA C M`)

#### Advanced Sampling Options
- `--tied_weights`: Use tied weights during prediction
- `--cfg`: Run classifier-free guidance
- `--sample_purity`: Enable purity sampling
- `--partial_flows`: Run partial flow matching
- `--t`: Forward diffusion time for partial flows (default: `0.5`)

### Examples

Process a single PDB file:
```bash
python sample.py \
  --pdb_path ./examples/6zht.pdb \
  --ckpt_path ./weights/mpnn_dataset/epoch\=66-step\=167499.pt \
  --num_predictions 8 \
  --output_root_dir ./output/
```

Process directory of PDB files:
```bash
python sample.py \
  --pdb_dir ./examples/ \
  --ckpt_path ./weights/mpnn_dataset/epoch\=66-step\=167499.pt \
  --num_predictions 8 \
  --output_root_dir ./output/
```

Keep chain A fixed and redesign all other residues using a higher sampling temperature (0.3), excluding cysteines, adding backbone noise with a standard deviation of 0.2, and generating 8 sequence samples:
```bash
python sample.py \
  --pdb_path ./examples/6zht.pdb \
  --chain_condition A \  
  --temp 0.3 \
  --omit_AA C \
  --noise_std 0.2 \
  --num_predictions 8 \
  --output_root_dir ./output/
```


Process multiple PDB files using JSON list (proteinmpnn style):
```bash
python sample.py \
  --json_path ./examples/files.json \
  --ckpt_path ./weights/mpnn_dataset/epoch\=66-step\=167499.pt \
  --num_predictions 8 \
  --output_root_dir ./output/
```

This repository is a modified version of OpenFold and incorporates components from MultiFlow, ProteinMPNN, and Protenix.

For any questions and concerns feel free to submit an issue









