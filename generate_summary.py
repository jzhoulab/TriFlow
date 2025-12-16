#!/usr/bin/env python3
"""
TriFlow Summary HTML Generator

Generates a standalone HTML summary file from TriFlow output directory containing:
- PDB structure files
- FASTA sequence files
- JSON probability files

The generated HTML includes:
- Interactive 3D structure viewer with shadows, outlines, SVG export
- MSA viewer for all sequences
- Sequence viewer with probability coloring
- Metadata display
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import re


def parse_fasta(fasta_path):
    """Parse FASTA file and return list of sequence records."""
    sequences = []
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append({
                        'header': current_header,
                        'sequence': ''.join(current_seq)
                    })
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_header is not None:
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_seq)
            })
    
    return sequences


def parse_pdb(pdb_path):
    """Read PDB file and return its contents as a string."""
    with open(pdb_path, 'r') as f:
        return f.read()


def parse_json_probs(json_path):
    """Parse JSON probability file and return metadata and residue probabilities."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def compute_secondary_structure(pdb_path):
    """Compute secondary structure using PyDSSP.
    
    PyDSSP is a simplified DSSP implementation that works with coordinates directly.
    Reference: https://github.com/ShintaroMinami/PyDSSP
    """
    try:
        import pydssp
        import numpy as np
        from Bio.PDB import PDBParser
        import warnings
        warnings.filterwarnings('ignore')
        
        # Parse PDB to get backbone coordinates (N, CA, C, O)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]
        
        coords = []
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard residue (not hetero)
                    try:
                        n = residue['N'].get_coord()
                        ca = residue['CA'].get_coord()
                        c = residue['C'].get_coord()
                        o = residue['O'].get_coord()
                        coords.append([n, ca, c, o])
                    except KeyError:
                        # Skip residues missing backbone atoms
                        continue
        
        if len(coords) == 0:
            return None
        
        # Convert to numpy array with shape (length, 4, 3)
        coord_array = np.array(coords, dtype=np.float32)
        
        # Run PyDSSP assignment
        # out_type='c3' gives C3 annotation: '-' (loop), 'H' (helix), 'E' (strand)
        dssp_result = pydssp.assign(coord_array, out_type='c3')
        
        # Convert to our format: H (helix), E (sheet), C (coil)
        ss_assignments = []
        for ss in dssp_result:
            if ss == 'H':
                ss_assignments.append('H')
            elif ss == 'E':
                ss_assignments.append('E')
            else:
                ss_assignments.append('C')
        
        return ss_assignments
    except ImportError as e:
        print(f"PyDSSP import error: {e}")
        return None
    except Exception as e:
        print(f"Secondary structure computation error: {e}")
        return None


def discover_output_files(output_dir):
    """Discover and group files in the output directory by prefix."""
    output_path = Path(output_dir)
    
    groups = defaultdict(lambda: {
        'pdb_files': [],
        'fasta_file': None,
        'json_files': [],
        'sequences': []
    })
    
    # Find backbone PDB files
    backbones_dir = output_path / 'backbones'
    if backbones_dir.exists():
        for pdb_file in sorted(backbones_dir.glob('*.pdb')):
            name = pdb_file.stem
            match = re.match(r'^(.+?)_\d+$', name)
            if match:
                prefix = match.group(1)
            else:
                prefix = name
            groups[prefix]['pdb_files'].append(pdb_file)
    
    # Find FASTA files
    seqs_dir = output_path / 'seqs'
    if seqs_dir.exists():
        for fasta_file in seqs_dir.glob('*.fa'):
            prefix = fasta_file.stem
            groups[prefix]['fasta_file'] = fasta_file
    
    # Find JSON probability files
    json_dir = output_path / 'json'
    if json_dir.exists():
        for json_file in sorted(json_dir.glob('*_unmasked_probs.json')):
            name = json_file.stem.replace('_unmasked_probs', '')
            match = re.match(r'^(.+?)_\d+$', name)
            if match:
                prefix = match.group(1)
            else:
                prefix = name
            groups[prefix]['json_files'].append(json_file)
    
    return dict(groups)


def generate_html(output_dir, output_file=None):
    """Generate standalone HTML summary file."""
    
    groups = discover_output_files(output_dir)
    
    if not groups:
        print("No valid files found in output directory.")
        return
    
    # Prepare data for HTML
    html_data = {
        'groups': {}
    }
    
    for prefix, files in groups.items():
        group_data = {
            'prefix': prefix,
            'structures': [],
            'sequences': [],
            'probabilities': {},
            'secondaryStructure': None
        }
        
        # Load first PDB file for each prefix
        if files['pdb_files']:
            pdb_path = files['pdb_files'][0]
            pdb_content = parse_pdb(pdb_path)
            group_data['structures'].append({
                'name': pdb_path.stem,
                'pdb': pdb_content
            })
            
            # Compute secondary structure using DSSP
            ss = compute_secondary_structure(pdb_path)
            if ss:
                group_data['secondaryStructure'] = ss
        
        # Load all sequences from FASTA
        if files['fasta_file']:
            sequences = parse_fasta(files['fasta_file'])
            group_data['sequences'] = sequences
        
        # Load probabilities from JSON files
        for json_file in files['json_files']:
            json_data = parse_json_probs(json_file)
            json_name = json_file.stem.replace('_unmasked_probs', '')
            group_data['probabilities'][json_name] = json_data
        
        html_data['groups'][prefix] = group_data
    
    # Generate HTML
    html_content = generate_html_content(html_data)
    
    # Determine output file path
    if output_file is None:
        output_file = Path(output_dir) / 'summary.html'
    else:
        output_file = Path(output_file)
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Summary HTML generated: {output_file}")
    return output_file


def generate_html_content(data):
    """Generate the HTML content string."""
    
    # Prepare JavaScript data
    js_data = json.dumps(data, indent=2)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TriFlow Summary</title>
    <style>
{STYLES}
    </style>
</head>
<body>
    <div class="container">
        <div id="content">
            <!-- Content will be generated by JavaScript -->
        </div>
    </div>
    
    <script>
    // Application data
    const triflowData = {js_data};
    
    // 3D Structure Viewer Class with full py2Dmol-style features
{STRUCTURE_VIEWER_JS}

    // MSA Viewer Class
{MSA_VIEWER_JS}

    // Sequence Viewer Class
{SEQUENCE_VIEWER_JS}
    
    // Initialize application
{APP_JS}
    </script>
</body>
</html>'''
    
    return html


STYLES = '''
:root {
    --color-primary: #6366f1;
    --color-primary-hover: #4f46e5;
    --color-primary-dark: #4338ca;
    --color-success: #10b981;
    --color-danger: #ef4444;
    --color-warning: #f59e0b;
    --color-gray-50: #f9fafb;
    --color-gray-100: #f3f4f6;
    --color-gray-200: #e5e7eb;
    --color-gray-300: #d1d5db;
    --color-gray-400: #9ca3af;
    --color-gray-500: #6b7280;
    --color-gray-600: #4b5563;
    --color-gray-700: #374151;
    --color-gray-800: #1f2937;
    --color-gray-900: #111827;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    min-height: 100vh;
    padding: 20px;
}

.container {
    max-width: 1600px;
    margin: 0 auto;
}

/* Summary header */
.summary-header {
    background: rgba(255, 255, 255, 0.98);
    border-radius: var(--radius-lg);
    padding: 16px 24px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-md);
}

.summary-info {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    color: var(--color-gray-600);
    font-size: 0.9rem;
}

.summary-stat strong {
    color: var(--color-primary);
}

.summary-divider {
    color: var(--color-gray-300);
}

.summary-files {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.summary-files-label {
    font-size: 0.85rem;
    color: var(--color-gray-500);
    margin-right: 4px;
}

.summary-file-tag {
    background: var(--color-gray-100);
    color: var(--color-gray-700);
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid var(--color-gray-200);
    cursor: pointer;
    transition: all 0.15s ease;
}

.summary-file-tag:hover {
    background: var(--color-primary);
    color: white;
    border-color: var(--color-primary);
}

.group-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-lg);
}

.group-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 2px solid var(--color-gray-200);
}

.group-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--color-gray-800);
}

.group-badge {
    background: var(--color-primary);
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
}

.content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: stretch;
}

@media (max-width: 1200px) {
    .content-grid {
        grid-template-columns: 1fr;
    }
}

.panel {
    background: var(--color-gray-50);
    border: 1px solid var(--color-gray-200);
    border-radius: var(--radius-md);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.panel-full {
    grid-column: 1 / -1;
}

.panel-header {
    background: var(--color-gray-100);
    padding: 12px 16px;
    border-bottom: 1px solid var(--color-gray-200);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}

.panel-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--color-gray-700);
}

.panel-body {
    padding: 16px;
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
}

/* Structure Viewer */
.structure-viewer-container {
    position: relative;
    width: 100%;
    background: #ffffff;
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--color-gray-200);
}

.structure-canvas {
    width: 100%;
    height: 450px;
    cursor: grab;
    display: block;
}

.structure-canvas:active {
    cursor: grabbing;
}

/* MSA Viewer */
.msa-container {
    background: white;
    border-radius: var(--radius-sm);
    overflow: hidden;
}

.msa-canvas {
    width: 100%;
    height: 300px;
    display: block;
    cursor: default;
}

.msa-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    background: var(--color-gray-100);
    border-bottom: 1px solid var(--color-gray-200);
}

.msa-mode-btn {
    padding: 5px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid var(--color-gray-300);
    background: white;
    color: var(--color-gray-600);
    transition: all 0.2s;
}

.msa-mode-btn:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
}

.msa-mode-btn.active {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: white;
}

/* Sequence Viewer */
.sequence-viewer {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
}

.sequence-entry {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 10px;
    background: white;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-gray-200);
    min-height: 0;
    overflow: hidden;
}

.sequence-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    flex-shrink: 0;
}

.color-mode-controls {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    margin-bottom: 8px;
    flex-shrink: 0;
}

.btn-xs {
    padding: 3px 6px;
    font-size: 0.65rem;
    border-radius: var(--radius-sm);
    font-weight: 500;
    cursor: pointer;
    border: 1px solid var(--color-gray-300);
    transition: all 0.15s;
}

.sequence-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--color-gray-700);
}

.sequence-meta {
    font-size: 0.75rem;
    color: var(--color-gray-500);
}

.sequence-content {
    font-family: 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
    font-size: 11px;
    line-height: 1.6;
    word-break: break-all;
    background: var(--color-gray-50);
    padding: 8px;
    border-radius: var(--radius-sm);
    flex: 1;
    overflow-y: auto;
    min-height: 100px;
}

.residue {
    display: inline;
    padding: 1px 2px;
    border-radius: 2px;
    transition: all 0.15s;
    cursor: pointer;
}

.residue:hover {
    transform: scale(1.2);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.residue.selected {
    outline: 2px solid #F59E0B;
    outline-offset: -1px;
    filter: brightness(0.7);
    box-shadow: 0 0 4px rgba(245, 158, 11, 0.5);
}

/* Metadata Panel */
.metadata-panel {
    margin-top: 20px;
}

.metadata-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    flex-shrink: 0;
}

.metadata-item {
    background: white;
    padding: 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-gray-200);
}

.metadata-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--color-gray-500);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.metadata-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--color-gray-800);
    word-break: break-all;
}

/* Controls */
.viewer-controls {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
    align-items: center;
}

.control-group {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: var(--color-gray-100);
    border-radius: var(--radius-sm);
}

.control-label {
    font-size: 0.75rem;
    color: var(--color-gray-600);
    font-weight: 500;
}

.btn {
    padding: 6px 12px;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
}

.btn-sm {
    padding: 4px 8px;
    font-size: 0.7rem;
}

.btn-primary {
    background: var(--color-primary);
    color: white;
}

.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.btn-secondary {
    background: var(--color-gray-200);
    color: var(--color-gray-700);
}

.btn-secondary:hover {
    background: var(--color-gray-300);
}

.btn-secondary.active {
    background: var(--color-success);
    color: white;
}

/* Checkbox styled as toggle */
.toggle-wrapper {
    display: flex;
    align-items: center;
    gap: 6px;
}

.toggle {
    position: relative;
    width: 36px;
    height: 20px;
    background: var(--color-gray-300);
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s;
}

.toggle.active {
    background: var(--color-success);
}

.toggle::after {
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: all 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}

.toggle.active::after {
    left: 18px;
}

/* Sequence selector */
.sequence-selector-container {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 12px;
    align-items: flex-start;
    flex-shrink: 0;
}

.sequence-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.sequence-group-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 2px 6px;
    border-radius: 4px;
    width: fit-content;
}

.reference-label {
    background: #FEF3C7;
    color: #92400E;
}

.designed-label {
    background: #DBEAFE;
    color: #1E40AF;
}

.sequence-selector {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}

.sequence-tab {
    padding: 5px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid var(--color-gray-300);
    background: white;
    color: var(--color-gray-600);
    transition: all 0.2s;
}

.sequence-tab:hover {
    border-color: var(--color-primary);
    color: var(--color-primary);
}

.sequence-tab.active {
    background: var(--color-primary);
    border-color: var(--color-primary);
    color: white;
}

.sequence-tab.reference-tab {
    border-color: #F59E0B;
}

.sequence-tab.reference-tab:hover {
    border-color: #D97706;
    color: #D97706;
}

.sequence-tab.reference-tab.active {
    background: #F59E0B;
    border-color: #F59E0B;
    color: white;
}

.sequence-tab.designed-tab {
    border-color: var(--color-gray-300);
}

/* Sequence type badges */
.sequence-name-container {
    display: flex;
    align-items: center;
    gap: 8px;
}

.sequence-type-badge {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 2px 8px;
    border-radius: 4px;
}

.reference-badge {
    background: #FEF3C7;
    color: #92400E;
}

.designed-badge {
    background: #DBEAFE;
    color: #1E40AF;
}

.sequence-entry.reference-entry {
    border-left: 3px solid #F59E0B;
    padding-left: 12px;
}

.sequence-entry.designed-entry {
    border-left: 3px solid var(--color-primary);
    padding-left: 12px;
}

/* Color legend */
.color-legend {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    padding: 8px;
    background: var(--color-gray-50);
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    flex-shrink: 0;
}

.legend-gradient {
    display: none;
}

.legend-blocks {
    display: flex;
    gap: 2px;
    align-items: center;
}

.legend-block {
    display: flex;
    align-items: center;
    gap: 3px;
    font-size: 0.7rem;
}

.legend-color {
    width: 16px;
    height: 14px;
    border-radius: 2px;
}

.legend-color-orange { background: #FF7E45; }
.legend-color-yellow { background: #FFDB12; }
.legend-color-lightblue { background: #57CAF9; }
.legend-color-darkblue { background: #0053D7; }

.legend-label {
    color: var(--color-gray-600);
}

select {
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-gray-300);
    font-size: 0.8rem;
    background: white;
    cursor: pointer;
}

select:focus {
    outline: none;
    border-color: var(--color-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

input[type="range"] {
    width: 80px;
    cursor: pointer;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--color-gray-100);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--color-gray-400);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--color-gray-500);
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.group-container {
    animation: fadeIn 0.3s ease-out;
}

/* Stats info */
.viewer-info {
    font-size: 0.7rem;
    color: var(--color-gray-500);
    margin-left: auto;
}

/* Slider controls */
.slider-control {
    display: flex;
    align-items: center;
    gap: 6px;
}

.slider-value {
    font-size: 0.7rem;
    color: var(--color-gray-600);
    min-width: 30px;
    text-align: center;
}
'''


STRUCTURE_VIEWER_JS = '''
// py2Dmol-style Structure Viewer
// Matches the rendering style of py2Dmol viewer-mol.js

class StructureViewer {
    constructor(canvas, pdbContent, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.atoms = [];
        this.chains = [];
        this.segments = [];  // Bond segments between consecutive CA atoms
        this.rotationMatrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        this.zoom = 1.0;
        this.colorMode = options.colorMode || 'chain';
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.autoRotate = false;
        this.animationId = null;
        
        // py2Dmol-style options
        this.shadowEnabled = options.shadow !== false;
        this.outlineMode = options.outline || 'full'; // 'none', 'partial', 'full'
        this.lineWidth = options.width || 3.0;
        this.depthEnabled = options.depth || false;
        this.relativeOutlineWidth = 3.0;
        this.shadowIntensity = 0.95;
        
        // Constants matching py2Dmol
        this.SHADOW_CUTOFF_MULTIPLIER = 2.0;
        this.TINT_CUTOFF_MULTIPLIER = 0.5;
        this.SHADOW_OFFSET_MULTIPLIER = 2.5;
        this.TINT_OFFSET_MULTIPLIER = 2.5;
        this.MAX_SHADOW_SUM = 12;
        this.REF_LENGTH = 3.8;  // Typical CA-CA distance
        
        // Cached data
        this.maxExtent = 0;
        this.center = { x: 0, y: 0, z: 0 };
        
        // Conservation data (set externally)
        this.conservationScores = null;
        
        // Probability data (set externally)
        this.probabilityScores = null;
        
        // Selected sequence (for charge/hydrophobicity coloring)
        this.selectedSequence = null;
        
        // Charge classification (1-letter codes)
        this.positiveCharge = new Set(['R', 'K', 'H']);
        this.negativeCharge = new Set(['D', 'E']);
        
        // Hydrophobicity scale (Kyte-Doolittle) - 1-letter codes
        this.hydrophobicity = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
            'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
            'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        };
        
        // Secondary structure assignments (from DSSP or computed from geometry)
        this.secondaryStructure = options.secondaryStructure || [];
        
        // Selection ranges for highlighting (supports multiple discontiguous selections)
        this.selectionRanges = null;  // Array of { start: number, end: number } or null
        
        this.parsePDB(pdbContent);
        
        // If no DSSP data provided, compute from geometry as fallback
        if (this.secondaryStructure.length === 0) {
            this.assignSecondaryStructure();
        }
        
        this.setupEvents();
        this.render();
    }
    
    parsePDB(pdbContent) {
        const lines = pdbContent.split('\\n');
        let currentChain = null;
        let chainIndex = -1;
        
        for (const line of lines) {
            if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
                const atomName = line.substring(12, 16).trim();
                
                // Only take CA atoms for backbone visualization
                if (atomName === 'CA') {
                    const x = parseFloat(line.substring(30, 38));
                    const y = parseFloat(line.substring(38, 46));
                    const z = parseFloat(line.substring(46, 54));
                    const chain = line.substring(21, 22).trim() || 'A';
                    const resNum = parseInt(line.substring(22, 26));
                    const resName = line.substring(17, 20).trim();
                    const bfactor = parseFloat(line.substring(60, 66)) || 50;
                    
                    if (chain !== currentChain) {
                        currentChain = chain;
                        chainIndex++;
                        this.chains.push({ id: chain, startIdx: this.atoms.length });
                    }
                    
                    this.atoms.push({
                        x, y, z,
                        chain: chainIndex,
                        chainId: chain,
                        resNum,
                        resName,
                        bfactor
                    });
                }
            }
        }
        
        // Build segments between consecutive CA atoms in same chain
        for (let i = 1; i < this.atoms.length; i++) {
            if (this.atoms[i].chain === this.atoms[i-1].chain) {
                const a1 = this.atoms[i-1];
                const a2 = this.atoms[i];
                const dx = a2.x - a1.x;
                const dy = a2.y - a1.y;
                const dz = a2.z - a1.z;
                const len = Math.sqrt(dx*dx + dy*dy + dz*dz);
                this.segments.push({
                    idx1: i-1,
                    idx2: i,
                    len: len
                });
            }
        }
        
        // Calculate center
        if (this.atoms.length > 0) {
            let sumX = 0, sumY = 0, sumZ = 0;
            for (const atom of this.atoms) {
                sumX += atom.x;
                sumY += atom.y;
                sumZ += atom.z;
            }
            this.center = {
                x: sumX / this.atoms.length,
                y: sumY / this.atoms.length,
                z: sumZ / this.atoms.length
            };
            
            // Calculate max extent from center
            this.maxExtent = 0;
            for (const atom of this.atoms) {
                const dx = atom.x - this.center.x;
                const dy = atom.y - this.center.y;
                const dz = atom.z - this.center.z;
                const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                if (dist > this.maxExtent) this.maxExtent = dist;
            }
            if (this.maxExtent < 1) this.maxExtent = 1;
        }
    }
    
    assignSecondaryStructure() {
        // Simple secondary structure assignment based on CA-CA distances and angles
        // H = helix, E = sheet, C = coil
        this.secondaryStructure = new Array(this.atoms.length).fill('C');
        
        if (this.atoms.length < 4) return;
        
        // Calculate dihedral angles and distances for each residue
        for (let i = 2; i < this.atoms.length - 1; i++) {
            // Check if all 4 atoms are in the same chain
            if (this.atoms[i-2].chain !== this.atoms[i+1].chain) continue;
            if (this.atoms[i-1].chain !== this.atoms[i].chain) continue;
            
            // Get CA positions
            const ca0 = this.atoms[i-2];
            const ca1 = this.atoms[i-1];
            const ca2 = this.atoms[i];
            const ca3 = this.atoms[i+1];
            
            // Calculate CA(i-2) to CA(i+1) distance (characteristic of helix ~5.5Å, sheet ~6.5-13Å)
            const dx = ca3.x - ca0.x;
            const dy = ca3.y - ca0.y;
            const dz = ca3.z - ca0.z;
            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
            
            // Calculate approximate phi/psi from CA positions
            // Helix: dist ~5-6Å for 4 consecutive CAs
            // Sheet: dist ~10-13Å for 4 consecutive CAs
            
            if (dist < 6.5) {
                // Likely helix
                this.secondaryStructure[i-1] = 'H';
                this.secondaryStructure[i] = 'H';
            } else if (dist > 10.0 && dist < 14.0) {
                // Likely sheet
                this.secondaryStructure[i-1] = 'E';
                this.secondaryStructure[i] = 'E';
            }
        }
        
        // Smooth out isolated assignments (require at least 3 consecutive for helix, 2 for sheet)
        for (let i = 1; i < this.secondaryStructure.length - 1; i++) {
            const prev = this.secondaryStructure[i-1];
            const curr = this.secondaryStructure[i];
            const next = this.secondaryStructure[i+1];
            
            // Remove isolated helix assignments
            if (curr === 'H' && prev !== 'H' && next !== 'H') {
                this.secondaryStructure[i] = 'C';
            }
            // Remove isolated sheet assignments  
            if (curr === 'E' && prev !== 'E' && next !== 'E') {
                this.secondaryStructure[i] = 'C';
            }
        }
    }
    
    _getSecondaryStructureColor(index) {
        if (index >= this.secondaryStructure.length) {
            return { r: 180, g: 180, b: 180 };  // Gray for unknown
        }
        
        const ss = this.secondaryStructure[index];
        if (ss === 'H') {
            return { r: 233, g: 86, b: 120 };   // Pink/Red - helix
        } else if (ss === 'E') {
            return { r: 255, g: 204, b: 0 };    // Yellow - sheet
        } else {
            return { r: 180, g: 180, b: 180 };  // Gray - coil
        }
    }
    
    setupEvents() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
        });
        
        window.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'grab';
        });
        
        window.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            
            // Apply rotation using matrix multiplication (like py2Dmol)
            const rotY = this._rotationMatrixY(deltaX * 0.01);
            const rotX = this._rotationMatrixX(deltaY * 0.01);
            this.rotationMatrix = this._multiplyMatrices(rotY, this.rotationMatrix);
            this.rotationMatrix = this._multiplyMatrices(rotX, this.rotationMatrix);
            
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            
            this.render();
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.zoom *= e.deltaY > 0 ? 0.95 : 1.05;
            this.zoom = Math.max(0.3, Math.min(3, this.zoom));
            this.render();
        });
    }
    
    _rotationMatrixX(angle) {
        const c = Math.cos(angle), s = Math.sin(angle);
        return [[1, 0, 0], [0, c, -s], [0, s, c]];
    }
    
    _rotationMatrixY(angle) {
        const c = Math.cos(angle), s = Math.sin(angle);
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
    }
    
    _multiplyMatrices(a, b) {
        const r = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                for (let k = 0; k < 3; k++) {
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return r;
    }
    
    _applyMatrix(m, x, y, z) {
        return {
            x: m[0][0] * x + m[0][1] * y + m[0][2] * z,
            y: m[1][0] * x + m[1][1] * y + m[1][2] * z,
            z: m[2][0] * x + m[2][1] * y + m[2][2] * z
        };
    }
    
    // HSV to RGB conversion (matching py2Dmol)
    _hsvToRgb(h, s, v) {
        const c = v * s;
        const x = c * (1 - Math.abs((h / 60) % 2 - 1));
        const m = v - c;
        let r, g, b;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        return {
            r: Math.round((r + m) * 255),
            g: Math.round((g + m) * 255),
            b: Math.round((b + m) * 255)
        };
    }
    
    // Rainbow color: N-term (blue, 240) to C-term (red, 0)
    _getRainbowColor(index, total) {
        if (total <= 1) return this._hsvToRgb(240, 1.0, 1.0);
        const normalized = index / (total - 1);
        const hue = 240 * (1 - normalized);  // 240 (blue) -> 0 (red)
        return this._hsvToRgb(hue, 1.0, 1.0);
    }
    
    // Chain colors (PyMOL-style)
    _getChainColor(chainIndex) {
        const pymolColors = [
            {r: 51, g: 255, b: 51},   // green
            {r: 0, g: 255, b: 255},    // cyan
            {r: 255, g: 51, b: 204},   // magenta
            {r: 255, g: 255, b: 0},    // yellow
            {r: 255, g: 153, b: 153},  // salmon
            {r: 229, g: 229, b: 229},  // white
            {r: 127, g: 127, b: 255},  // slate
            {r: 255, g: 127, b: 0}     // orange
        ];
        return pymolColors[chainIndex % pymolColors.length];
    }
    
    // pLDDT color: 50 (red) to 90 (blue)
    _getPlddtColor(plddt) {
        const min = 50, max = 90;
        let normalized = (plddt - min) / (max - min);
        normalized = Math.max(0, Math.min(1, normalized));
        const hue = 240 * normalized;  // 0 (red) -> 240 (blue)
        return this._hsvToRgb(hue, 1.0, 1.0);
    }
    
    getColor(index, total, chainIdx, bfactor, resName) {
        if (this.colorMode === 'chain') {
            return this._getChainColor(chainIdx);
        } else if (this.colorMode === 'plddt' || this.colorMode === 'bfactor') {
            return this._getPlddtColor(bfactor);
        } else if (this.colorMode === 'probability') {
            return this._getProbabilityColor(index, chainIdx);
        } else if (this.colorMode === 'charge') {
            return this._getChargeColor(index, chainIdx);
        } else if (this.colorMode === 'hydrophobicity') {
            return this._getHydrophobicityColor(index, chainIdx);
        } else if (this.colorMode === 'conservation') {
            return this._getConservationColor(index);
        } else if (this.colorMode === 'secondary') {
            return this._getSecondaryStructureColor(index);
        } else {
            // Rainbow coloring based on sequence position
            return this._getRainbowColor(index, total);
        }
    }
    
    _getChargeColor(index, chainIdx) {
        // Get amino acid from selected sequence, fall back to chain color if not available
        const aa = this._getAAFromSequence(index);
        if (!aa) {
            return this._getChainColor(chainIdx);
        }
        
        if (this.positiveCharge.has(aa)) {
            return { r: 59, g: 130, b: 246 };  // Blue - positive
        } else if (this.negativeCharge.has(aa)) {
            return { r: 239, g: 68, b: 68 };   // Red - negative
        }
        return { r: 180, g: 180, b: 180 };     // Gray - neutral
    }
    
    _getHydrophobicityColor(index, chainIdx) {
        // Get amino acid from selected sequence, fall back to chain color if not available
        const aa = this._getAAFromSequence(index);
        if (!aa) {
            return this._getChainColor(chainIdx);
        }
        
        const value = this.hydrophobicity[aa];
        if (value === undefined) return { r: 180, g: 180, b: 180 };
        
        // Map from -4.5 to 4.5 to color gradient
        const normalized = (value + 4.5) / 9.0;  // 0 to 1
        
        if (normalized > 0.5) {
            // Hydrophobic - orange to brown
            const intensity = (normalized - 0.5) * 2;
            return {
                r: Math.round(234 - intensity * 50),
                g: Math.round(179 - intensity * 100),
                b: Math.round(8 + intensity * 20)
            };
        } else {
            // Hydrophilic - light blue to blue
            const intensity = (0.5 - normalized) * 2;
            return {
                r: Math.round(147 - intensity * 88),
                g: Math.round(197 - intensity * 60),
                b: Math.round(253 - intensity * 9)
            };
        }
    }
    
    _getAAFromSequence(index) {
        if (!this.selectedSequence) return null;
        // Remove chain separators and get AA at index
        const cleanSeq = this.selectedSequence.replace(/:/g, '');
        if (index >= cleanSeq.length) return null;
        return cleanSeq[index].toUpperCase();
    }
    
    setSequence(sequence) {
        this.selectedSequence = sequence;
        // Re-render if in a mode that depends on sequence
        if (this.colorMode === 'charge' || this.colorMode === 'hydrophobicity') {
            this.render();
        }
    }
    
    _getConservationColor(index) {
        if (!this.conservationScores || index >= this.conservationScores.length) {
            return { r: 180, g: 180, b: 180 };
        }
        
        const score = this.conservationScores[index];
        
        if (score >= 0.9) {
            return { r: 124, g: 58, b: 237 };   // Purple - highly conserved
        } else if (score >= 0.7) {
            return { r: 167, g: 139, b: 250 };  // Light purple
        } else if (score >= 0.5) {
            return { r: 196, g: 181, b: 253 };  // Very light purple
        } else {
            return { r: 180, g: 180, b: 180 };  // Gray - variable
        }
    }
    
    _getProbabilityColor(index, chainIdx) {
        // If no probability data, fall back to chain coloring
        if (!this.probabilityScores || index >= this.probabilityScores.length) {
            return this._getChainColor(chainIdx);
        }
        
        const prob = this.probabilityScores[index];
        if (prob === null || prob === undefined) {
            return this._getChainColor(chainIdx);
        }
        
        // AlphaFold pLDDT color scale (same as sequence viewer):
        // <= 0.5: Orange (#FF7E45)
        // > 0.5 and <= 0.7: Yellow (#FFDB12)
        // > 0.7 and <= 0.9: Light Blue (#57CAF9)
        // > 0.9: Dark Blue (#0053D7)
        if (prob <= 0.5) {
            return { r: 255, g: 126, b: 69 };   // Orange - low confidence
        } else if (prob <= 0.7) {
            return { r: 255, g: 219, b: 18 };   // Yellow - medium confidence
        } else if (prob <= 0.9) {
            return { r: 87, g: 202, b: 249 };   // Light Blue - high confidence
        } else {
            return { r: 0, g: 83, b: 215 };     // Dark Blue - very high confidence
        }
    }
    
    setConservationScores(scores) {
        this.conservationScores = scores;
        if (this.colorMode === 'conservation') {
            this.render();
        }
    }
    
    setProbabilityScores(scores) {
        this.probabilityScores = scores;
        if (this.colorMode === 'probability') {
            this.render();
        }
    }
    
    setSelectionRanges(ranges) {
        this.selectionRanges = ranges;
        this.render();
    }
    
    // Legacy method for backward compatibility
    setSelectionRange(range) {
        this.selectionRanges = range ? [range] : null;
        this.render();
    }
    
    isResidueSelected(index) {
        if (!this.selectionRanges || this.selectionRanges.length === 0) return false;
        for (const range of this.selectionRanges) {
            if (index >= range.start && index <= range.end) {
                return true;
            }
        }
        return false;
    }
    
    // Shadow/tint calculation matching py2Dmol
    _calculateShadowTint(s1, s2, len1, len2) {
        const avgLen = (len1 + len2) * 0.5 || this.REF_LENGTH;
        const shadow_cutoff = avgLen * this.SHADOW_CUTOFF_MULTIPLIER;
        const tint_cutoff = avgLen * this.TINT_CUTOFF_MULTIPLIER;
        const shadow_offset = this.REF_LENGTH * this.SHADOW_OFFSET_MULTIPLIER;
        const tint_offset = this.REF_LENGTH * this.TINT_OFFSET_MULTIPLIER;
        
        const max_cutoff = shadow_cutoff + shadow_offset;
        const max_cutoff_sq = max_cutoff * max_cutoff;
        
        const dx = s1.x - s2.x;
        const dy = s1.y - s2.y;
        const dist2D_sq = dx * dx + dy * dy;
        
        if (dist2D_sq > max_cutoff_sq) {
            return { shadow: 0, tint: 0 };
        }
        
        let shadow = 0;
        let tint = 0;
        
        const dz = s1.z - s2.z;
        const dist3D_sq = dist2D_sq + dz * dz;
        
        // Shadow calculation
        if (dist3D_sq < max_cutoff_sq) {
            const shadow_cutoff_sq = shadow_cutoff * shadow_cutoff;
            const alpha = 2.0;
            shadow = shadow_cutoff_sq / (shadow_cutoff_sq + dist3D_sq * alpha);
        }
        
        // Tint calculation
        const tint_max_cutoff = tint_cutoff + tint_offset;
        const tint_max_cutoff_sq = tint_max_cutoff * tint_max_cutoff;
        if (dist2D_sq < tint_max_cutoff_sq) {
            const tint_cutoff_sq = tint_cutoff * tint_cutoff;
            const alpha = 2.0;
            tint = tint_cutoff_sq / (tint_cutoff_sq + dist2D_sq * alpha);
        }
        
        return { shadow, tint };
    }
    
    render() {
        const ctx = this.ctx;
        const displayWidth = parseInt(this.canvas.style.width) || this.canvas.width;
        const displayHeight = parseInt(this.canvas.style.height) || this.canvas.height;
        
        // Clear with WHITE background (py2Dmol style)
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, displayWidth, displayHeight);
        
        if (this.atoms.length === 0 || this.segments.length === 0) {
            ctx.fillStyle = '#666';
            ctx.font = '16px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No structure data', displayWidth/2, displayHeight/2);
            return;
        }
        
        const m = this.rotationMatrix;
        const c = this.center;
        const n = this.segments.length;
        
        // Rotate all atoms
        const rotated = this.atoms.map(atom => {
            const subX = atom.x - c.x;
            const subY = atom.y - c.y;
            const subZ = atom.z - c.z;
            return this._applyMatrix(m, subX, subY, subZ);
        });
        
        // Build segment data with midpoints
        const segData = this.segments.map((seg, idx) => {
            const start = rotated[seg.idx1];
            const end = rotated[seg.idx2];
            return {
                idx: idx,
                idx1: seg.idx1,
                idx2: seg.idx2,
                len: seg.len,
                x: (start.x + end.x) * 0.5,
                y: (start.y + end.y) * 0.5,
                z: (start.z + end.z) * 0.5,
                start: start,
                end: end
            };
        });
        
        // Calculate z-range for normalization
        let zMin = Infinity, zMax = -Infinity;
        for (const s of segData) {
            if (s.z < zMin) zMin = s.z;
            if (s.z > zMax) zMax = s.z;
        }
        
        // Calculate zNorm for depth effects
        const zRange = zMax - zMin || 1;
        const zNorm = segData.map(s => (s.z - zMin) / zRange);
        
        // Sort segments back to front
        const order = segData.map((s, i) => ({ idx: i, z: s.z }))
            .sort((a, b) => a.z - b.z)
            .map(item => item.idx);
        
        // Build segment order map for endpoint rounding detection
        const segmentOrderMap = new Map();
        for (let i = 0; i < order.length; i++) {
            segmentOrderMap.set(order[i], i);
        }
        
        // Build position-to-segments maps for detecting outer endpoints
        const positionToSegmentsStartingAt = new Map();
        const positionToSegmentsEndingAt = new Map();
        for (let segIdx = 0; segIdx < n; segIdx++) {
            const seg = this.segments[segIdx];
            
            if (!positionToSegmentsStartingAt.has(seg.idx1)) {
                positionToSegmentsStartingAt.set(seg.idx1, []);
            }
            positionToSegmentsStartingAt.get(seg.idx1).push(segIdx);
            
            if (!positionToSegmentsEndingAt.has(seg.idx2)) {
                positionToSegmentsEndingAt.set(seg.idx2, []);
            }
            positionToSegmentsEndingAt.get(seg.idx2).push(segIdx);
        }
        
        // Pre-compute which endpoints should have rounded caps
        // An endpoint is "outer" if it's at the start/end of a chain (no other segment connects there)
        const segmentEndpointRounding = new Map();
        for (let segIdx = 0; segIdx < n; segIdx++) {
            const seg = this.segments[segIdx];
            const currentOrderIdx = segmentOrderMap.get(segIdx);
            
            const shouldRoundEndpoint = (positionIndex) => {
                // Get all segments connected to this position
                const segmentsEnding = positionToSegmentsEndingAt.get(positionIndex) || [];
                const segmentsStarting = positionToSegmentsStartingAt.get(positionIndex) || [];
                const allSegments = [...segmentsEnding, ...segmentsStarting];
                
                // If only one segment uses this position, it's an outer endpoint
                if (allSegments.length <= 1) {
                    return true;
                }
                
                // For multi-segment junctions, only round if this segment is rendered first (furthest back)
                let lowestOrderIdx = currentOrderIdx;
                for (const otherSegIdx of allSegments) {
                    const otherOrderIdx = segmentOrderMap.get(otherSegIdx);
                    if (otherOrderIdx !== undefined && otherOrderIdx < lowestOrderIdx) {
                        lowestOrderIdx = otherOrderIdx;
                    }
                }
                return currentOrderIdx === lowestOrderIdx;
            };
            
            segmentEndpointRounding.set(segIdx, {
                hasOuterStart: shouldRoundEndpoint(seg.idx1),
                hasOuterEnd: shouldRoundEndpoint(seg.idx2)
            });
        }
        
        // Calculate shadows and tints (py2Dmol style)
        const shadows = new Float32Array(n).fill(1.0);
        const tints = new Float32Array(n).fill(1.0);
        
        if (this.shadowEnabled) {
            for (let i_idx = order.length - 1; i_idx >= 0; i_idx--) {
                const i = order[i_idx];
                let shadowSum = 0;
                let maxTint = 0;
                const s1 = segData[i];
                
                for (let j_idx = i_idx + 1; j_idx < order.length; j_idx++) {
                    const j = order[j_idx];
                    
                    if (shadowSum >= this.MAX_SHADOW_SUM) break;
                    
                    const s2 = segData[j];
                    const { shadow, tint } = this._calculateShadowTint(s1, s2, s1.len, s2.len);
                    shadowSum = Math.min(shadowSum + shadow, this.MAX_SHADOW_SUM);
                    maxTint = Math.max(maxTint, tint);
                }
                
                shadows[i] = Math.pow(this.shadowIntensity, shadowSum);
                tints[i] = 1 - maxTint;
            }
        }
        
        // Calculate scale (py2Dmol style)
        const padding = 0.9;
        const scaleX = (displayWidth * padding) / (this.maxExtent * 2);
        const scaleY = (displayHeight * padding) / (this.maxExtent * 2);
        const baseScale = Math.min(scaleX, scaleY);
        const scale = baseScale * this.zoom;
        const baseLineWidthPixels = this.lineWidth * scale;
        
        const centerX = displayWidth / 2;
        const centerY = displayHeight / 2;
        
        // Pre-compute colors for all segments
        const colors = this.segments.map((seg, idx) => {
            const atom = this.atoms[seg.idx2];
            return this.getColor(seg.idx2, this.atoms.length, atom.chain, atom.bfactor, atom.resName);
        });
        
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Draw segments in order (back to front)
        for (const idx of order) {
            const seg = this.segments[idx];
            const s = segData[idx];
            
            // Get base color
            let { r, g, b } = colors[idx];
            r /= 255; g /= 255; b /= 255;
            
            // Check if this segment is in the selection range
            const isSelected = this.isResidueSelected(seg.idx2);
            
            // Apply selection darkening (make selected residues darker/more saturated)
            if (isSelected) {
                // Darken the color significantly for selection
                const darkenFactor = 0.5;
                r *= darkenFactor;
                g *= darkenFactor;
                b *= darkenFactor;
            }
            
            // Apply shadow/tint effects (py2Dmol style)
            const zNormVal = zNorm[idx];
            
            if (this.shadowEnabled) {
                const tintFactor = this.depthEnabled
                    ? (0.50 * zNormVal + 0.50 * tints[idx]) / 3
                    : (0.50 * tints[idx]) / 3;
                r = r + (1 - r) * tintFactor;
                g = g + (1 - g) * tintFactor;
                b = b + (1 - b) * tintFactor;
                
                const shadowFactor = this.depthEnabled
                    ? (0.20 + 0.25 * zNormVal + 0.55 * shadows[idx])
                    : (0.20 + 0.80 * shadows[idx]);
                r *= shadowFactor;
                g *= shadowFactor;
                b *= shadowFactor;
            } else if (this.depthEnabled) {
                const depthFactor = 0.70 + 0.30 * zNormVal;
                r *= depthFactor;
                g *= depthFactor;
                b *= depthFactor;
            }
            
            // Project to screen
            const x1 = centerX + s.start.x * scale;
            const y1 = centerY - s.start.y * scale;
            const x2 = centerX + s.end.x * scale;
            const y2 = centerY - s.end.y * scale;
            
            const currentLineWidth = Math.max(0.5, baseLineWidthPixels);
            
            // Convert to integers for color string
            const r_int = Math.round(r * 255);
            const g_int = Math.round(g * 255);
            const b_int = Math.round(b * 255);
            const color = `rgb(${r_int},${g_int},${b_int})`;
            
            // Get pre-computed endpoint rounding flags
            const endpointFlags = segmentEndpointRounding.get(idx) || { hasOuterStart: false, hasOuterEnd: false };
            const hasOuterStart = endpointFlags.hasOuterStart;
            const hasOuterEnd = endpointFlags.hasOuterEnd;
            
            if (this.outlineMode === 'none') {
                // No outline - just draw the main line with round caps
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = color;
                ctx.lineWidth = currentLineWidth;
                ctx.lineCap = 'round';
                ctx.stroke();
            } else if (this.outlineMode === 'partial') {
                // Partial outline - butt caps only (no rounded caps on outline)
                const gapFillerColor = `rgb(${Math.round(r_int * 0.7)},${Math.round(g_int * 0.7)},${Math.round(b_int * 0.7)})`;
                const totalOutlineWidth = currentLineWidth + this.relativeOutlineWidth;
                
                // Pass 1: Outline with butt caps
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = gapFillerColor;
                ctx.lineWidth = totalOutlineWidth;
                ctx.lineCap = 'butt';
                ctx.stroke();
                
                // Pass 2: Main colored line with round caps
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = color;
                ctx.lineWidth = currentLineWidth;
                ctx.lineCap = 'round';
                ctx.stroke();
            } else {
                // Full outline - rounded caps at OUTER endpoints only (py2Dmol style)
                const gapFillerColor = `rgb(${Math.round(r_int * 0.7)},${Math.round(g_int * 0.7)},${Math.round(b_int * 0.7)})`;
                const totalOutlineWidth = currentLineWidth + this.relativeOutlineWidth;
                
                // Pass 1: Outline with butt caps
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = gapFillerColor;
                ctx.lineWidth = totalOutlineWidth;
                ctx.lineCap = 'butt';
                ctx.stroke();
                
                // Add rounded caps at outer endpoints only
                const outlineRadius = totalOutlineWidth / 2;
                if (hasOuterStart) {
                    ctx.beginPath();
                    ctx.arc(x1, y1, outlineRadius, 0, Math.PI * 2);
                    ctx.fillStyle = gapFillerColor;
                    ctx.fill();
                }
                if (hasOuterEnd) {
                    ctx.beginPath();
                    ctx.arc(x2, y2, outlineRadius, 0, Math.PI * 2);
                    ctx.fillStyle = gapFillerColor;
                    ctx.fill();
                }
                
                // Pass 2: Main colored line with round caps
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = color;
                ctx.lineWidth = currentLineWidth;
                ctx.lineCap = 'round';
                ctx.stroke();
            }
        }
    }
    
    setColorMode(mode) {
        this.colorMode = mode;
        this.render();
    }
    
    setShadow(enabled) {
        this.shadowEnabled = enabled;
        this.render();
    }
    
    setOutline(mode) {
        this.outlineMode = mode;
        this.render();
    }
    
    setLineWidth(width) {
        this.lineWidth = width;
        this.render();
    }
    
    setDepth(enabled) {
        this.depthEnabled = enabled;
        this.render();
    }
    
    toggleAutoRotate() {
        this.autoRotate = !this.autoRotate;
        if (this.autoRotate) {
            this.animate();
        } else if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        return this.autoRotate;
    }
    
    animate() {
        if (!this.autoRotate) return;
        const rot = this._rotationMatrixY(0.0075);  // 50% slower (was 0.015)
        this.rotationMatrix = this._multiplyMatrices(rot, this.rotationMatrix);
        this.render();
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    resetView() {
        this.rotationMatrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        this.zoom = 1.0;
        this.autoRotate = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.render();
    }
    
    exportSVG() {
        // Export as true vector SVG with all rendering effects
        const displayWidth = parseInt(this.canvas.style.width) || this.canvas.width;
        const displayHeight = parseInt(this.canvas.style.height) || this.canvas.height;
        
        let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${displayWidth}" height="${displayHeight}" viewBox="0 0 ${displayWidth} ${displayHeight}">\\n`;
        svg += `  <rect width="${displayWidth}" height="${displayHeight}" fill="#ffffff"/>\\n`;
        
        if (this.atoms.length === 0 || this.segments.length === 0) {
            svg += '</svg>';
            return svg;
        }
        
        const m = this.rotationMatrix;
        const c = this.center;
        const n = this.segments.length;
        
        // Rotate all atoms
        const rotated = this.atoms.map(atom => {
            const subX = atom.x - c.x;
            const subY = atom.y - c.y;
            const subZ = atom.z - c.z;
            return this._applyMatrix(m, subX, subY, subZ);
        });
        
        // Build segment data with midpoints
        const segData = this.segments.map((seg, idx) => {
            const start = rotated[seg.idx1];
            const end = rotated[seg.idx2];
            return {
                idx: idx,
                idx1: seg.idx1,
                idx2: seg.idx2,
                len: seg.len,
                x: (start.x + end.x) * 0.5,
                y: (start.y + end.y) * 0.5,
                z: (start.z + end.z) * 0.5,
                start: start,
                end: end
            };
        });
        
        // Calculate z-range for normalization
        let zMin = Infinity, zMax = -Infinity;
        for (const s of segData) {
            if (s.z < zMin) zMin = s.z;
            if (s.z > zMax) zMax = s.z;
        }
        
        const zRange = zMax - zMin || 1;
        const zNorm = segData.map(s => (s.z - zMin) / zRange);
        
        // Sort segments back to front
        const order = segData.map((s, i) => ({ idx: i, z: s.z }))
            .sort((a, b) => a.z - b.z)
            .map(item => item.idx);
        
        // Build segment order map and endpoint rounding
        const segmentOrderMap = new Map();
        for (let i = 0; i < order.length; i++) {
            segmentOrderMap.set(order[i], i);
        }
        
        const positionToSegmentsStartingAt = new Map();
        const positionToSegmentsEndingAt = new Map();
        for (let segIdx = 0; segIdx < n; segIdx++) {
            const seg = this.segments[segIdx];
            if (!positionToSegmentsStartingAt.has(seg.idx1)) {
                positionToSegmentsStartingAt.set(seg.idx1, []);
            }
            positionToSegmentsStartingAt.get(seg.idx1).push(segIdx);
            if (!positionToSegmentsEndingAt.has(seg.idx2)) {
                positionToSegmentsEndingAt.set(seg.idx2, []);
            }
            positionToSegmentsEndingAt.get(seg.idx2).push(segIdx);
        }
        
        const segmentEndpointRounding = new Map();
        for (let segIdx = 0; segIdx < n; segIdx++) {
            const seg = this.segments[segIdx];
            const currentOrderIdx = segmentOrderMap.get(segIdx);
            
            const shouldRoundEndpoint = (positionIndex) => {
                const segmentsEnding = positionToSegmentsEndingAt.get(positionIndex) || [];
                const segmentsStarting = positionToSegmentsStartingAt.get(positionIndex) || [];
                const allSegments = [...segmentsEnding, ...segmentsStarting];
                
                if (allSegments.length <= 1) return true;
                
                let lowestOrderIdx = currentOrderIdx;
                for (const otherSegIdx of allSegments) {
                    const otherOrderIdx = segmentOrderMap.get(otherSegIdx);
                    if (otherOrderIdx !== undefined && otherOrderIdx < lowestOrderIdx) {
                        lowestOrderIdx = otherOrderIdx;
                    }
                }
                return currentOrderIdx === lowestOrderIdx;
            };
            
            segmentEndpointRounding.set(segIdx, {
                hasOuterStart: shouldRoundEndpoint(seg.idx1),
                hasOuterEnd: shouldRoundEndpoint(seg.idx2)
            });
        }
        
        // Calculate shadows and tints
        const shadows = new Float32Array(n).fill(1.0);
        const tints = new Float32Array(n).fill(1.0);
        
        if (this.shadowEnabled) {
            for (let i_idx = order.length - 1; i_idx >= 0; i_idx--) {
                const i = order[i_idx];
                let shadowSum = 0;
                let maxTint = 0;
                const s1 = segData[i];
                
                for (let j_idx = i_idx + 1; j_idx < order.length; j_idx++) {
                    const j = order[j_idx];
                    if (shadowSum >= this.MAX_SHADOW_SUM) break;
                    
                    const s2 = segData[j];
                    const { shadow, tint } = this._calculateShadowTint(s1, s2, s1.len, s2.len);
                    shadowSum = Math.min(shadowSum + shadow, this.MAX_SHADOW_SUM);
                    maxTint = Math.max(maxTint, tint);
                }
                
                shadows[i] = Math.pow(this.shadowIntensity, shadowSum);
                tints[i] = 1 - maxTint;
            }
        }
        
        // Calculate scale
        const padding = 0.9;
        const scaleX = (displayWidth * padding) / (this.maxExtent * 2);
        const scaleY = (displayHeight * padding) / (this.maxExtent * 2);
        const baseScale = Math.min(scaleX, scaleY);
        const scale = baseScale * this.zoom;
        const baseLineWidthPixels = this.lineWidth * scale;
        
        const centerX = displayWidth / 2;
        const centerY = displayHeight / 2;
        
        // Pre-compute colors
        const colors = this.segments.map((seg, idx) => {
            const atom = this.atoms[seg.idx2];
            return this.getColor(seg.idx2, this.atoms.length, atom.chain, atom.bfactor, atom.resName);
        });
        
        // Draw segments in order (back to front)
        for (const idx of order) {
            const seg = this.segments[idx];
            const s = segData[idx];
            
            // Get base color
            let { r, g, b } = colors[idx];
            r /= 255; g /= 255; b /= 255;
            
            // Check if this segment is in the selection range
            const isSelected = this.isResidueSelected(seg.idx2);
            
            // Apply selection darkening
            if (isSelected) {
                const darkenFactor = 0.5;
                r *= darkenFactor;
                g *= darkenFactor;
                b *= darkenFactor;
            }
            
            // Apply shadow/tint effects
            const zNormVal = zNorm[idx];
            
            if (this.shadowEnabled) {
                const tintFactor = this.depthEnabled
                    ? (0.50 * zNormVal + 0.50 * tints[idx]) / 3
                    : (0.50 * tints[idx]) / 3;
                r = r + (1 - r) * tintFactor;
                g = g + (1 - g) * tintFactor;
                b = b + (1 - b) * tintFactor;
                
                const shadowFactor = this.depthEnabled
                    ? (0.20 + 0.25 * zNormVal + 0.55 * shadows[idx])
                    : (0.20 + 0.80 * shadows[idx]);
                r *= shadowFactor;
                g *= shadowFactor;
                b *= shadowFactor;
            } else if (this.depthEnabled) {
                const depthFactor = 0.70 + 0.30 * zNormVal;
                r *= depthFactor;
                g *= depthFactor;
                b *= depthFactor;
            }
            
            // Project to screen
            const x1 = centerX + s.start.x * scale;
            const y1 = centerY - s.start.y * scale;
            const x2 = centerX + s.end.x * scale;
            const y2 = centerY - s.end.y * scale;
            
            const currentLineWidth = Math.max(0.5, baseLineWidthPixels);
            
            // Convert to integers for color string
            const r_int = Math.round(r * 255);
            const g_int = Math.round(g * 255);
            const b_int = Math.round(b * 255);
            const color = `rgb(${r_int},${g_int},${b_int})`;
            
            // Get endpoint rounding flags
            const endpointFlags = segmentEndpointRounding.get(idx) || { hasOuterStart: false, hasOuterEnd: false };
            const hasOuterStart = endpointFlags.hasOuterStart;
            const hasOuterEnd = endpointFlags.hasOuterEnd;
            
            if (this.outlineMode === 'none') {
                // No outline - just draw the main line
                svg += `  <line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${color}" stroke-width="${currentLineWidth.toFixed(2)}" stroke-linecap="round"/>\\n`;
            } else if (this.outlineMode === 'partial') {
                // Partial outline - butt caps only
                const gapFillerColor = `rgb(${Math.round(r_int * 0.7)},${Math.round(g_int * 0.7)},${Math.round(b_int * 0.7)})`;
                const totalOutlineWidth = currentLineWidth + this.relativeOutlineWidth;
                
                // Outline with butt caps
                svg += `  <line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${gapFillerColor}" stroke-width="${totalOutlineWidth.toFixed(2)}" stroke-linecap="butt"/>\\n`;
                // Main line with round caps
                svg += `  <line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${color}" stroke-width="${currentLineWidth.toFixed(2)}" stroke-linecap="round"/>\\n`;
            } else {
                // Full outline - rounded caps at outer endpoints only
                const gapFillerColor = `rgb(${Math.round(r_int * 0.7)},${Math.round(g_int * 0.7)},${Math.round(b_int * 0.7)})`;
                const totalOutlineWidth = currentLineWidth + this.relativeOutlineWidth;
                
                // Outline with butt caps
                svg += `  <line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${gapFillerColor}" stroke-width="${totalOutlineWidth.toFixed(2)}" stroke-linecap="butt"/>\\n`;
                
                // Add rounded caps at outer endpoints
                const outlineRadius = totalOutlineWidth / 2;
                if (hasOuterStart) {
                    svg += `  <circle cx="${x1.toFixed(2)}" cy="${y1.toFixed(2)}" r="${outlineRadius.toFixed(2)}" fill="${gapFillerColor}"/>\\n`;
                }
                if (hasOuterEnd) {
                    svg += `  <circle cx="${x2.toFixed(2)}" cy="${y2.toFixed(2)}" r="${outlineRadius.toFixed(2)}" fill="${gapFillerColor}"/>\\n`;
                }
                
                // Main line with round caps
                svg += `  <line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="${color}" stroke-width="${currentLineWidth.toFixed(2)}" stroke-linecap="round"/>\\n`;
            }
        }
        
        svg += '</svg>';
        return svg;
    }
    
    downloadSVG(filename = 'structure.svg') {
        const svg = this.exportSVG();
        const blob = new Blob([svg], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    downloadPNG(filename = 'structure.png') {
        const url = this.canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    getAtomCount() {
        return this.atoms.length;
    }
    
    getChainCount() {
        return this.chains.length;
    }
    
    // Highlight a specific residue and show info tooltip
    highlightResidue(residueIndex, residueName, probability) {
        this.highlightedResidue = residueIndex;
        this.highlightInfo = {
            index: residueIndex,
            name: residueName,
            probability: probability
        };
        this.render();
        this.drawHighlightOverlay();
    }
    
    // Clear highlight
    clearHighlight() {
        this.highlightedResidue = null;
        this.highlightInfo = null;
        this.render();
    }
    
    // Draw highlight overlay on top of rendered structure
    drawHighlightOverlay() {
        if (this.highlightedResidue === null || this.highlightedResidue >= this.atoms.length) return;
        
        const ctx = this.ctx;
        const displayWidth = parseInt(this.canvas.style.width) || this.canvas.width;
        const displayHeight = parseInt(this.canvas.style.height) || this.canvas.height;
        
        // Get the atom position
        const atom = this.atoms[this.highlightedResidue];
        if (!atom) return;
        
        // Transform the atom position
        const m = this.rotationMatrix;
        const c = this.center;
        const subX = atom.x - c.x;
        const subY = atom.y - c.y;
        const subZ = atom.z - c.z;
        const rotated = this._applyMatrix(m, subX, subY, subZ);
        
        // Calculate scale
        const padding = 0.9;
        const scaleX = (displayWidth * padding) / (this.maxExtent * 2);
        const scaleY = (displayHeight * padding) / (this.maxExtent * 2);
        const baseScale = Math.min(scaleX, scaleY);
        const scale = baseScale * this.zoom;
        
        const centerX = displayWidth / 2;
        const centerY = displayHeight / 2;
        
        // Project to screen
        const screenX = centerX + rotated.x * scale;
        const screenY = centerY - rotated.y * scale;
        
        // Draw highlight circle - small like a laser pointer
        const highlightRadius = 4;
        ctx.beginPath();
        ctx.arc(screenX, screenY, highlightRadius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 0, 0.9)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 200, 0, 1)';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw info tooltip in bottom right corner
        if (this.highlightInfo) {
            const padding = 10;
            const fontSize = 12;
            const lineHeight = 16;
            const bgPadding = 8;
            
            ctx.font = `${fontSize}px monospace`;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            
            // Build tooltip lines
            const lines = [
                `Residue: ${this.highlightInfo.index + 1}`,
                `Type: ${this.highlightInfo.name}`
            ];
            if (this.highlightInfo.probability !== null && this.highlightInfo.probability !== undefined) {
                lines.push(`Prob: ${this.highlightInfo.probability.toFixed(3)}`);
            }
            
            // Measure text
            const textMetrics = lines.map(line => ctx.measureText(line));
            const maxWidth = Math.max(...textMetrics.map(m => m.width));
            const totalHeight = lines.length * lineHeight;
            const bgWidth = maxWidth + bgPadding * 2;
            const bgHeight = totalHeight + bgPadding * 2;
            
            // Position in bottom right corner
            const x = displayWidth - padding;
            const y = displayHeight - padding;
            
            // Draw background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            ctx.beginPath();
            ctx.roundRect(x - bgWidth, y - bgHeight, bgWidth, bgHeight, 4);
            ctx.fill();
            
            // Draw text
            ctx.fillStyle = '#ffffff';
            lines.forEach((line, i) => {
                ctx.fillText(line, x - bgPadding, y - bgPadding - (lines.length - 1 - i) * lineHeight);
            });
        }
    }
}
'''


MSA_VIEWER_JS = '''
// MSA Viewer - Alignment with MSA, PSSM, and Logo modes (inspired by py2Dmol)
class MSAViewer {
    constructor(container, sequences) {
        this.container = container;
        this.sequences = sequences;
        this.scrollLeft = 0;
        this.scrollTop = 0;
        this.charWidth = 14;
        this.rowHeight = 16;
        this.nameWidth = 100;
        this.maxSeqLen = 0;
        this.viewMode = 'msa'; // 'msa', 'pssm', or 'logo'
        this.useBitScore = true; // For logo mode: true = bits, false = probability
        
        // Dayhoff amino acid groupings (same as py2Dmol viewer-msa.js)
        // 8 groups with specific colors matching py2Dmol
        this.DAYHOFF_GROUP_DEFINITIONS = [
            { name: 'group1', label: 'KR', aminoAcids: ['K', 'R'], color: { r: 212, g: 68, b: 43 } },      // #d4442b - Red
            { name: 'group2', label: 'AFILMVW', aminoAcids: ['A', 'F', 'I', 'L', 'M', 'V', 'W'], color: { r: 61, g: 126, b: 223 } }, // #3d7edf - Blue
            { name: 'group3', label: 'NQST', aminoAcids: ['N', 'Q', 'S', 'T'], color: { r: 96, g: 201, b: 65 } },  // #60c941 - Green
            { name: 'group4', label: 'HY', aminoAcids: ['H', 'Y'], color: { r: 83, g: 177, b: 178 } },     // #53b1b2 - Teal
            { name: 'group5', label: 'C', aminoAcids: ['C'], color: { r: 217, g: 133, b: 130 } },          // #d98582 - Pink
            { name: 'group6', label: 'DE', aminoAcids: ['D', 'E'], color: { r: 189, g: 85, b: 198 } },     // #bd55c6 - Purple
            { name: 'group7', label: 'P', aminoAcids: ['P'], color: { r: 204, g: 204, b: 65 } },           // #cccc41 - Yellow
            { name: 'group8', label: 'G', aminoAcids: ['G'], color: { r: 219, g: 157, b: 91 } }            // #db9d5b - Orange
        ];
        
        // Build DAYHOFF_COLORS and DAYHOFF_GROUPS from definitions
        this.DAYHOFF_COLORS = {};
        this.DAYHOFF_GROUPS = {};
        this.DAYHOFF_GROUP_DEFINITIONS.forEach(group => {
            this.DAYHOFF_COLORS[group.name] = group.color;
            group.aminoAcids.forEach(aa => {
                this.DAYHOFF_GROUPS[aa] = group.name;
            });
        });
        this.DAYHOFF_COLORS.gap = { r: 200, g: 200, b: 200 };
        this.DAYHOFF_COLORS.other = { r: 150, g: 150, b: 150 };
        
        // Amino acids ordered by Dayhoff groups for PSSM (same order as py2Dmol)
        this.AMINO_ACIDS_ORDERED = this.DAYHOFF_GROUP_DEFINITIONS.flatMap(group => group.aminoAcids);
        
        // Calculate group boundaries dynamically
        this.DAYHOFF_GROUP_BOUNDARIES = [];
        let currentIndex = 0;
        for (let i = 1; i < this.DAYHOFF_GROUP_DEFINITIONS.length; i++) {
            currentIndex += this.DAYHOFF_GROUP_DEFINITIONS[i - 1].aminoAcids.length;
            this.DAYHOFF_GROUP_BOUNDARIES.push(currentIndex);
        }
        
        // Standard amino acid background frequencies (for information content calculation)
        this.BACKGROUND_FREQUENCIES = {
            'A': 0.082, 'R': 0.057, 'N': 0.044, 'D': 0.053, 'C': 0.017,
            'Q': 0.040, 'E': 0.062, 'G': 0.072, 'H': 0.022, 'I': 0.052,
            'L': 0.090, 'K': 0.057, 'M': 0.024, 'F': 0.039, 'P': 0.051,
            'S': 0.069, 'T': 0.058, 'W': 0.013, 'Y': 0.032, 'V': 0.066
        };
        
        // Unique color for each amino acid (Taylor color scheme - used in MSA mode)
        this.aaColors = {
            'A': { r: 204, g: 255, b: 0 },    // Lime green
            'R': { r: 0, g: 0, b: 255 },      // Blue
            'N': { r: 0, g: 220, b: 220 },    // Cyan
            'D': { r: 255, g: 0, b: 0 },      // Red
            'C': { r: 255, g: 255, b: 0 },    // Yellow
            'Q': { r: 0, g: 220, b: 220 },    // Cyan
            'E': { r: 255, g: 0, b: 102 },    // Magenta-red
            'G': { r: 255, g: 153, b: 0 },    // Orange
            'H': { r: 0, g: 102, b: 255 },    // Light blue
            'I': { r: 102, g: 255, b: 0 },    // Yellow-green
            'L': { r: 51, g: 255, b: 0 },     // Green
            'K': { r: 102, g: 0, b: 255 },    // Purple
            'M': { r: 0, g: 255, b: 0 },      // Green
            'F': { r: 0, g: 102, b: 153 },    // Teal
            'P': { r: 220, g: 150, b: 130 },  // Salmon
            'S': { r: 255, g: 51, b: 0 },     // Orange-red
            'T': { r: 255, g: 102, b: 0 },    // Dark orange
            'W': { r: 0, g: 204, b: 153 },    // Teal-green
            'Y': { r: 51, g: 204, b: 204 },   // Cyan-teal
            'V': { r: 153, g: 255, b: 0 },    // Yellow-green
            '-': { r: 255, g: 255, b: 255 },  // White for gaps
            '.': { r: 255, g: 255, b: 255 },  // White for gaps
            'X': { r: 180, g: 180, b: 180 }   // Gray for unknown
        };
        
        // Glyph metrics cache for scaled letter rendering
        this.glyphMetricsCache = new Map();
        
        // Selection ranges for highlighting (supports multiple discontiguous selections)
        this.selectionRanges = null;  // Array of { start: number, end: number } or null
        
        // Parse sequences
        this.parsedSeqs = this.sequences.map(s => {
            const parts = s.header.split(',');
            return {
                name: parts[0].trim(),
                sequence: s.sequence.replace(/:/g, '') // Remove chain separators
            };
        });
        
        this.maxSeqLen = Math.max(...this.parsedSeqs.map(s => s.sequence.length));
        
        // Pre-compute position frequencies for PSSM and Logo
        this.frequencies = this.computePositionFrequencies();
        
        this.render();
    }
    
    getBackgroundFrequency(aa) {
        if (!aa || aa === '-' || aa === 'X') return 0;
        return this.BACKGROUND_FREQUENCIES[aa.toUpperCase()] || (1 / 20);
    }
    
    getDayhoffColor(aa) {
        if (!aa || aa === '-' || aa === 'X') return this.DAYHOFF_COLORS.gap;
        const group = this.DAYHOFF_GROUPS[aa.toUpperCase()];
        if (group) return this.DAYHOFF_COLORS[group];
        return this.DAYHOFF_COLORS.other;
    }
    
    // Get Dayhoff color with slight variation for each amino acid within a group
    getDayhoffColorWithVariation(aa) {
        if (!aa || aa === '-' || aa === 'X') return this.DAYHOFF_COLORS.gap;
        
        const aaUpper = aa.toUpperCase();
        const group = this.DAYHOFF_GROUPS[aaUpper];
        
        if (!group) return this.DAYHOFF_COLORS.other;
        
        const baseColor = this.DAYHOFF_COLORS[group];
        
        // Find which group definition this belongs to and position within group
        let groupDef = null;
        let positionInGroup = -1;
        
        for (const def of this.DAYHOFF_GROUP_DEFINITIONS) {
            if (def.name === group) {
                groupDef = def;
                positionInGroup = def.aminoAcids.indexOf(aaUpper);
                break;
            }
        }
        
        // If single amino acid in group, no variation needed
        if (!groupDef || groupDef.aminoAcids.length === 1) {
            return baseColor;
        }
        
        // Apply slight variation based on position in group
        // Vary brightness/saturation slightly to distinguish within group
        const numInGroup = groupDef.aminoAcids.length;
        const variationFactor = (positionInGroup / (numInGroup - 1)) * 0.3 - 0.15; // Range: -0.15 to +0.15
        
        // Convert to HSL, adjust lightness, convert back
        const r = baseColor.r / 255;
        const g = baseColor.g / 255;
        const b = baseColor.b / 255;
        
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;
        
        if (max === min) {
            h = s = 0; // achromatic
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            
            switch (max) {
                case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
                case g: h = ((b - r) / d + 2) / 6; break;
                case b: h = ((r - g) / d + 4) / 6; break;
            }
        }
        
        // Adjust lightness
        l = Math.max(0.1, Math.min(0.9, l + variationFactor));
        
        // Convert back to RGB
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        
        let newR, newG, newB;
        if (s === 0) {
            newR = newG = newB = l;
        } else {
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            newR = hue2rgb(p, q, h + 1/3);
            newG = hue2rgb(p, q, h);
            newB = hue2rgb(p, q, h - 1/3);
        }
        
        return {
            r: Math.round(newR * 255),
            g: Math.round(newG * 255),
            b: Math.round(newB * 255)
        };
    }
    
    getAAColor(aa) {
        if (!aa) return { r: 255, g: 255, b: 255 };
        const color = this.aaColors[aa.toUpperCase()];
        return color || { r: 180, g: 180, b: 180 };
    }
    
    // Get contrasting text color (black or white) based on background
    getTextColor(bgColor) {
        const luminance = (0.299 * bgColor.r + 0.587 * bgColor.g + 0.114 * bgColor.b) / 255;
        return luminance > 0.5 ? '#000000' : '#ffffff';
    }
    
    // Compute position frequencies for PSSM and Logo visualization
    computePositionFrequencies() {
        const frequencies = [];
        const numSequences = this.parsedSeqs.length;
        
        for (let pos = 0; pos < this.maxSeqLen; pos++) {
            const posFreq = {};
            // Initialize all amino acids to 0
            for (const aa of this.AMINO_ACIDS_ORDERED) {
                posFreq[aa] = 0;
            }
            
            let validCount = 0;
            for (let seqIdx = 0; seqIdx < numSequences; seqIdx++) {
                const seq = this.parsedSeqs[seqIdx].sequence;
                if (pos < seq.length) {
                    const aa = seq[pos].toUpperCase();
                    if (this.AMINO_ACIDS_ORDERED.includes(aa)) {
                        posFreq[aa]++;
                        validCount++;
                    }
                }
            }
            
            // Normalize to frequencies
            if (validCount > 0) {
                for (const aa of this.AMINO_ACIDS_ORDERED) {
                    posFreq[aa] /= validCount;
                }
            }
            
            frequencies.push(posFreq);
        }
        
        return frequencies;
    }
    
    // Get glyph metrics for scaled letter rendering (WebLogo-style) - same as py2Dmol
    getGlyphMetrics(ctx, ch) {
        const key = ch;
        if (this.glyphMetricsCache.has(key)) return this.glyphMetricsCache.get(key);
        
        ctx.save();
        ctx.font = 'bold 100px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        const m = ctx.measureText(ch);
        ctx.restore();
        
        // Use nullish coalescing (??) to properly handle 0 values - same as py2Dmol
        const left = m.actualBoundingBoxLeft ?? 0;
        const right = m.actualBoundingBoxRight ?? (m.width ?? 100);
        const ascent = m.actualBoundingBoxAscent ?? 80;
        const desc = m.actualBoundingBoxDescent ?? 20;
        
        // Special handling for Q: its tail can extend beyond normal bounds
        // Use measured width for Q to get more accurate bounds - same as py2Dmol
        let glyphWidth = (left + right) || (m.width || 100);
        if (ch === 'Q' || ch === 'q') {
            glyphWidth = m.width || 100;
        }
        
        const metrics = {
            left,
            width: glyphWidth || 1,
            ascent,
            descent: desc,
            height: (ascent + desc) || 1
        };
        this.glyphMetricsCache.set(key, metrics);
        return metrics;
    }
    
    // Draw scaled letter (WebLogo-style) with optional clipping
    drawScaledLetter(ctx, ch, x, yBottom, w, h, color, clipRect = null) {
        if (h <= 0 || w <= 0) return;
        const g = this.getGlyphMetrics(ctx, ch);
        const sx = w / g.width;
        const sy = h / g.height;
        
        // Adjust vertical position upward for all letters to keep descenders visible
        // This ensures letters with parts extending below baseline (Q, S, G, etc.) stay within bounds
        const yOffset = g.descent * sy * 1.0;
        
        ctx.save();
        // Apply clipRect if provided to ensure no gaps between letters
        if (clipRect) {
            ctx.beginPath();
            ctx.rect(clipRect.x, clipRect.y, clipRect.w, clipRect.h);
            ctx.clip();
        }
        ctx.translate(x + g.left * sx, yBottom - yOffset);
        ctx.scale(sx, sy);
        ctx.fillStyle = color;
        ctx.font = 'bold 100px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(ch, 0, 0);
        ctx.restore();
    }
    
    setViewMode(mode) {
        this.viewMode = mode;
        this.render();
    }
    
    toggleBitScore() {
        this.useBitScore = !this.useBitScore;
        this.render();
    }
    
    setSelectionRanges(ranges) {
        this.selectionRanges = ranges;
        this.render();
    }
    
    // Legacy method for backward compatibility
    setSelectionRange(range) {
        this.selectionRanges = range ? [range] : null;
        this.render();
    }
    
    isPositionSelected(pos) {
        if (!this.selectionRanges || this.selectionRanges.length === 0) return false;
        for (const range of this.selectionRanges) {
            if (pos >= range.start && pos <= range.end) {
                return true;
            }
        }
        return false;
    }
    
    render() {
        this.container.innerHTML = '';
        
        // Header with mode toggle
        const header = document.createElement('div');
        header.className = 'msa-header';
        header.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-weight: 600; color: var(--color-gray-700);">Sequence Alignment</span>
                <button class="msa-mode-btn ${this.viewMode === 'msa' ? 'active' : ''}" data-mode="msa">MSA</button>
                <button class="msa-mode-btn ${this.viewMode === 'pssm' ? 'active' : ''}" data-mode="pssm">PSSM</button>
                <button class="msa-mode-btn ${this.viewMode === 'logo' ? 'active' : ''}" data-mode="logo">Logo</button>
                ${this.viewMode === 'logo' ? `
                    <span style="margin-left: 10px; display: flex; gap: 4px;">
                        <button class="msa-mode-btn ${!this.useBitScore ? 'active' : ''}" id="probToggle">Probability</button>
                        <button class="msa-mode-btn ${this.useBitScore ? 'active' : ''}" id="bitsToggle">Bits</button>
                    </span>
                ` : ''}
            </div>
            <span style="margin-left: auto; font-size: 0.75rem; color: var(--color-gray-500);">
                ${this.parsedSeqs.length} sequences, ${this.maxSeqLen} positions
            </span>
        `;
        this.container.appendChild(header);
        
        // Add mode toggle event listeners
        header.querySelectorAll('.msa-mode-btn[data-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setViewMode(e.target.dataset.mode);
            });
        });
        
        // Add probability/bits toggle listeners
        const probToggle = header.querySelector('#probToggle');
        const bitsToggle = header.querySelector('#bitsToggle');
        if (probToggle) {
            probToggle.addEventListener('click', () => {
                if (this.useBitScore) {
                    this.useBitScore = false;
                    this.render();
                }
            });
        }
        if (bitsToggle) {
            bitsToggle.addEventListener('click', () => {
                if (!this.useBitScore) {
                    this.useBitScore = true;
                    this.render();
                }
            });
        }
        
        // Canvas container
        const canvasContainer = document.createElement('div');
        canvasContainer.style.position = 'relative';
        canvasContainer.style.overflow = 'auto';
        canvasContainer.style.maxHeight = '400px';
        this.container.appendChild(canvasContainer);
        
        const canvas = document.createElement('canvas');
        canvas.className = 'msa-canvas';
        canvasContainer.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        
        if (this.viewMode === 'pssm') {
            this.renderPSSM(canvas, ctx, dpr);
        } else if (this.viewMode === 'logo') {
            this.renderLogo(canvas, ctx, dpr);
        } else {
            this.renderMSA(canvas, ctx, dpr);
        }
    }
    
    renderMSA(canvas, ctx, dpr) {
        const numSeqs = this.parsedSeqs.length;
        const totalWidth = this.nameWidth + this.maxSeqLen * this.charWidth + 20;
        const totalHeight = numSeqs * this.rowHeight + 30;
        
        canvas.width = totalWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.width = totalWidth + 'px';
        canvas.style.height = totalHeight + 'px';
        ctx.scale(dpr, dpr);
        
        // Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, totalWidth, totalHeight);
        
        // Draw position numbers
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        for (let i = 0; i < this.maxSeqLen; i += 10) {
            const x = this.nameWidth + i * this.charWidth + this.charWidth / 2;
            ctx.fillText(String(i + 1), x, 10);
        }
        
        // Draw sequences
        ctx.font = '11px monospace';
        ctx.textBaseline = 'middle';
        
        for (let seqIdx = 0; seqIdx < numSeqs; seqIdx++) {
            const seq = this.parsedSeqs[seqIdx];
            const y = 18 + seqIdx * this.rowHeight + this.rowHeight / 2;
            
            // Sequence name
            ctx.fillStyle = '#333';
            ctx.textAlign = 'right';
            const displayName = seq.name.length > 10 ? seq.name.substring(0, 10) + '..' : seq.name;
            ctx.fillText(displayName, this.nameWidth - 4, y);
            
            // Sequence characters
            ctx.textAlign = 'center';
            for (let pos = 0; pos < seq.sequence.length; pos++) {
                const aa = seq.sequence[pos].toUpperCase();
                const x = this.nameWidth + pos * this.charWidth + this.charWidth / 2;
                const isSelected = this.isPositionSelected(pos);
                
                // Background color (Dayhoff with variations)
                let bgColor = this.getDayhoffColorWithVariation(aa);
                
                // Darken if selected
                if (isSelected) {
                    bgColor = {
                        r: Math.round(bgColor.r * 0.5),
                        g: Math.round(bgColor.g * 0.5),
                        b: Math.round(bgColor.b * 0.5)
                    };
                }
                
                ctx.fillStyle = `rgb(${bgColor.r}, ${bgColor.g}, ${bgColor.b})`;
                ctx.fillRect(this.nameWidth + pos * this.charWidth, y - this.rowHeight/2 + 1, this.charWidth, this.rowHeight - 2);
                
                // Character (contrasting text color)
                ctx.fillStyle = this.getTextColor(bgColor);
                ctx.fillText(aa, x, y);
            }
        }
        
        // Draw selection indicator bars at the top if there are selections
        if (this.selectionRanges && this.selectionRanges.length > 0) {
            ctx.fillStyle = 'rgba(251, 191, 36, 0.8)';  // Amber color
            
            for (const range of this.selectionRanges) {
                const startX = this.nameWidth + range.start * this.charWidth;
                const endX = this.nameWidth + (range.end + 1) * this.charWidth;
                const width = endX - startX;
                
                // Draw selection bar
                ctx.fillRect(startX, 12, width, 4);
            }
            
            // Draw selection range text for first range only to avoid clutter
            const firstRange = this.selectionRanges[0];
            const startX = this.nameWidth + firstRange.start * this.charWidth;
            ctx.fillStyle = '#92400E';
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'left';
            const rangeText = this.selectionRanges.length === 1 
                ? `${firstRange.start + 1}-${firstRange.end + 1}`
                : `${this.selectionRanges.length} selections`;
            ctx.fillText(rangeText, startX, 10);
        }
    }
    
    renderPSSM(canvas, ctx, dpr) {
        const NUM_AMINO_ACIDS = this.AMINO_ACIDS_ORDERED.length;
        const labelWidth = 25;  // Width for amino acid labels
        const boxWidth = 7;     // Width of each position box (half of normal)
        const aaRowHeight = 14; // Height of each amino acid row
        const tickRowHeight = 16;
        const queryRowHeight = 16;
        
        const heatmapWidth = this.maxSeqLen * boxWidth;
        const heatmapHeight = NUM_AMINO_ACIDS * aaRowHeight;
        
        const totalWidth = labelWidth + heatmapWidth + 20;
        const totalHeight = tickRowHeight + queryRowHeight + heatmapHeight + 20;
        
        canvas.width = totalWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.width = totalWidth + 'px';
        canvas.style.height = totalHeight + 'px';
        ctx.scale(dpr, dpr);
        
        // Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, totalWidth, totalHeight);
        
        const heatmapX = labelWidth;
        const heatmapY = tickRowHeight + queryRowHeight;
        
        // Draw tick marks (position numbers)
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        for (let i = 0; i < this.maxSeqLen; i += 10) {
            const x = heatmapX + i * boxWidth + boxWidth / 2;
            ctx.fillText(String(i + 1), x, 10);
        }
        
        // Draw query sequence (first sequence)
        if (this.parsedSeqs.length > 0) {
            const querySeq = this.parsedSeqs[0].sequence;
            const queryY = tickRowHeight;
            ctx.font = '10px monospace';
            ctx.textBaseline = 'middle';
            
            for (let pos = 0; pos < querySeq.length; pos++) {
                const aa = querySeq[pos].toUpperCase();
                const x = heatmapX + pos * boxWidth;
                const isSelected = this.isPositionSelected(pos);
                
                let color = this.getDayhoffColor(aa);
                
                // Darken if selected
                if (isSelected) {
                    color = {
                        r: Math.round(color.r * 0.5),
                        g: Math.round(color.g * 0.5),
                        b: Math.round(color.b * 0.5)
                    };
                }
                
                ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                ctx.fillRect(x, queryY, boxWidth, queryRowHeight);
                
                // Draw character if box is wide enough
                if (boxWidth >= 10) {
                    ctx.fillStyle = isSelected ? '#fff' : '#000';
                    ctx.textAlign = 'center';
                    ctx.fillText(aa, x + boxWidth / 2, queryY + queryRowHeight / 2);
                }
            }
        }
        
        // Draw amino acid labels (left side)
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        for (let i = 0; i < NUM_AMINO_ACIDS; i++) {
            const aa = this.AMINO_ACIDS_ORDERED[i];
            const y = heatmapY + i * aaRowHeight;
            
            // Background color for label
            const color = this.getDayhoffColor(aa);
            ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
            ctx.fillRect(0, y, labelWidth, aaRowHeight);
            
            // Amino acid letter
            ctx.fillStyle = '#000';
            ctx.fillText(aa, labelWidth / 2, y + aaRowHeight / 2);
        }
        
        // Draw heatmap
        for (let pos = 0; pos < this.frequencies.length; pos++) {
            const posData = this.frequencies[pos];
            const x = heatmapX + pos * boxWidth;
            const isSelected = this.isPositionSelected(pos);
            
            for (let i = 0; i < NUM_AMINO_ACIDS; i++) {
                const aa = this.AMINO_ACIDS_ORDERED[i];
                const probability = posData[aa] || 0;
                const y = heatmapY + i * aaRowHeight;
                
                // Color: white (0) to dark blue (1)
                const white = { r: 255, g: 255, b: 255 };
                const darkBlue = { r: 0, g: 0, b: 139 };
                let finalR = Math.round(white.r + (darkBlue.r - white.r) * probability);
                let finalG = Math.round(white.g + (darkBlue.g - white.g) * probability);
                let finalB = Math.round(white.b + (darkBlue.b - white.b) * probability);
                
                // Darken if selected
                if (isSelected) {
                    finalR = Math.round(finalR * 0.6);
                    finalG = Math.round(finalG * 0.6);
                    finalB = Math.round(finalB * 0.6);
                }
                
                ctx.fillStyle = `rgb(${finalR}, ${finalG}, ${finalB})`;
                ctx.fillRect(x, y, boxWidth, aaRowHeight);
            }
        }
        
        // Draw selection indicator bars at the top if there are selections
        if (this.selectionRanges && this.selectionRanges.length > 0) {
            ctx.fillStyle = 'rgba(251, 191, 36, 0.8)';
            
            for (const range of this.selectionRanges) {
                const startX = heatmapX + range.start * boxWidth;
                const endX = heatmapX + (range.end + 1) * boxWidth;
                const width = endX - startX;
                ctx.fillRect(startX, 2, width, 4);
            }
        }
        
        // Draw black boxes around wildtype (query sequence amino acids)
        if (this.parsedSeqs.length > 0) {
            const querySeq = this.parsedSeqs[0].sequence;
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1;
            
            for (let pos = 0; pos < querySeq.length; pos++) {
                const wildtypeAA = querySeq[pos].toUpperCase();
                const wildtypeIndex = this.AMINO_ACIDS_ORDERED.indexOf(wildtypeAA);
                
                if (wildtypeIndex >= 0) {
                    const x = heatmapX + pos * boxWidth;
                    const y = heatmapY + wildtypeIndex * aaRowHeight;
                    
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + boxWidth, y);
                    ctx.moveTo(x, y + aaRowHeight);
                    ctx.lineTo(x + boxWidth, y + aaRowHeight);
                    ctx.moveTo(x, y);
                    ctx.lineTo(x, y + aaRowHeight);
                    ctx.moveTo(x + boxWidth, y);
                    ctx.lineTo(x + boxWidth, y + aaRowHeight);
                    ctx.stroke();
                }
            }
        }
        
        // Draw group boundaries
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;
        for (const boundaryIdx of this.DAYHOFF_GROUP_BOUNDARIES) {
            const y = heatmapY + boundaryIdx * aaRowHeight;
            ctx.beginPath();
            ctx.moveTo(heatmapX, y);
            ctx.lineTo(heatmapX + heatmapWidth, y);
            ctx.stroke();
        }
    }
    
    renderLogo(canvas, ctx, dpr) {
        const charWidth = this.charWidth;
        const yAxisWidth = 45;
        const tickRowHeight = 16;
        const queryRowHeight = 16;
        const logoHeight = 150;
        const logoVerticalPadding = 12;
        
        const totalWidth = yAxisWidth + this.maxSeqLen * charWidth + 20;
        const totalHeight = logoVerticalPadding + logoHeight + queryRowHeight + tickRowHeight + 20;
        
        canvas.width = totalWidth * dpr;
        canvas.height = totalHeight * dpr;
        canvas.style.width = totalWidth + 'px';
        canvas.style.height = totalHeight + 'px';
        ctx.scale(dpr, dpr);
        
        // Background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, totalWidth, totalHeight);
        
        const logoX = yAxisWidth;
        const logoY = logoVerticalPadding;
        const queryY = logoY + logoHeight;
        const tickY = queryY + queryRowHeight;
        const effectiveLogoHeight = logoHeight; // Full height for logo
        
        // Compute logo data
        const logoData = [];
        let maxInfoContent = 0;
        
        if (this.useBitScore) {
            // Bit score mode: information content
            const positionInfoContents = [];
            
            for (let pos = 0; pos < this.frequencies.length; pos++) {
                const posFreq = this.frequencies[pos];
                let infoContent = 0;
                const contributions = {};
                
                for (const aa in posFreq) {
                    const freq = posFreq[aa];
                    if (freq > 0) {
                        const backgroundFreq = this.getBackgroundFrequency(aa);
                        if (backgroundFreq > 0) {
                            const contribution = freq * Math.log2(freq / backgroundFreq);
                            if (contribution > 0) {
                                infoContent += contribution;
                                contributions[aa] = contribution;
                            }
                        }
                    }
                }
                
                positionInfoContents.push({ infoContent, contributions });
                if (infoContent > maxInfoContent) {
                    maxInfoContent = infoContent;
                }
            }
            
            // Convert to letter heights - stack proportionally
            for (let pos = 0; pos < positionInfoContents.length; pos++) {
                const posInfo = positionInfoContents[pos];
                const infoContent = posInfo.infoContent;
                const contributions = posInfo.contributions;
                
                // Total stack height is proportional to information content
                const totalStackHeight = maxInfoContent > 0
                    ? (infoContent / maxInfoContent) * effectiveLogoHeight
                    : 0;
                
                const letterHeights = {};
                if (infoContent > 0) {
                    for (const aa in contributions) {
                        // Each letter's height is its proportion of the total stack
                        letterHeights[aa] = (contributions[aa] / infoContent) * totalStackHeight;
                    }
                }
                
                logoData.push({ infoContent, letterHeights });
            }
        } else {
            // Probability mode: frequencies fill full height
            for (let pos = 0; pos < this.frequencies.length; pos++) {
                const posFreq = this.frequencies[pos];
                const letterHeights = {};
                
                let freqSum = 0;
                for (const aa in posFreq) {
                    freqSum += posFreq[aa];
                }
                
                // Normalize frequencies to sum to 1.0, then scale to full logo height
                const normalizationFactor = freqSum > 0 ? 1 / freqSum : 1;
                for (const aa in posFreq) {
                    letterHeights[aa] = (posFreq[aa] * normalizationFactor) * effectiveLogoHeight;
                }
                
                logoData.push({ letterHeights });
            }
        }
        
        // Draw Y-axis
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, yAxisWidth, totalHeight);
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(yAxisWidth, logoY);
        ctx.lineTo(yAxisWidth, queryY);
        ctx.stroke();
        
        // Y-axis label
        const axisLabel = this.useBitScore ? 'Bits' : 'Probability';
        ctx.save();
        ctx.translate(15, (logoY + queryY) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillStyle = '#333';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(axisLabel, 0, 0);
        ctx.restore();
        
        // Y-axis ticks
        ctx.fillStyle = '#333';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        
        const tickValues = [];
        if (this.useBitScore) {
            const maxVal = maxInfoContent > 0 ? maxInfoContent : 1;
            tickValues.push({ value: 0, label: '0' });
            tickValues.push({ value: maxVal / 2, label: (maxVal / 2).toFixed(1) });
            tickValues.push({ value: maxVal, label: maxVal.toFixed(1) });
        } else {
            tickValues.push({ value: 0, label: '0.0' });
            tickValues.push({ value: 0.5, label: '0.5' });
            tickValues.push({ value: 1.0, label: '1.0' });
        }
        
        for (const tick of tickValues) {
            let yPos;
            if (this.useBitScore) {
                const maxVal = maxInfoContent > 0 ? maxInfoContent : 1;
                yPos = queryY - (tick.value / maxVal) * effectiveLogoHeight;
            } else {
                yPos = queryY - tick.value * effectiveLogoHeight;
            }
            
            ctx.fillText(tick.label, yAxisWidth - 8, yPos);
            ctx.beginPath();
            ctx.moveTo(yAxisWidth - 5, yPos);
            ctx.lineTo(yAxisWidth, yPos);
            ctx.stroke();
        }
        
        // Draw stacked logo - letters stacked from bottom up with NO gaps
        // Use clipping to ensure letters extend all the way to query row (same as py2Dmol)
        const scrollableAreaWidth = this.maxSeqLen * charWidth;
        const clipRect = { x: logoX, y: logoY, w: scrollableAreaWidth, h: queryY - logoY };
        
        for (let pos = 0; pos < logoData.length; pos++) {
            const x = logoX + pos * charWidth;
            const letterHeights = logoData[pos].letterHeights || {};
            const isSelected = this.isPositionSelected(pos);
            
            // Sort by height ascending (smallest first) - smallest at bottom, tallest at top
            const aas = Object.keys(letterHeights).sort((a, b) => letterHeights[a] - letterHeights[b]);
            
            // Start from bottom (queryY) and stack upward - NO gaps between letters
            let yOffset = queryY;
            
            for (const aa of aas) {
                const height = letterHeights[aa];
                // In probability mode, draw all letters; in bit mode, skip very small ones
                const isProbabilitiesMode = !this.useBitScore;
                const shouldDraw = isProbabilitiesMode ? true : height > 1;
                // Ensure minimum visible height for very small letters
                const drawHeight = isProbabilitiesMode && height > 0 && height < 0.5 ? 0.5 : height;
                
                if (shouldDraw && drawHeight > 0) {
                    // Use Dayhoff colors (same as py2Dmol Logo mode)
                    let color = this.getDayhoffColor(aa);
                    
                    // Darken if selected
                    if (isSelected) {
                        color = {
                            r: Math.round(color.r * 0.5),
                            g: Math.round(color.g * 0.5),
                            b: Math.round(color.b * 0.5)
                        };
                    }
                    
                    const colorStr = `rgb(${color.r}, ${color.g}, ${color.b})`;
                    
                    // Clip logo rendering to extend all the way to queryY (no gap) - same as py2Dmol
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(logoX, logoY, scrollableAreaWidth, queryY - logoY);
                    ctx.clip();
                    
                    // WebLogo-style letter mode: scale glyph bbox to fill full cell
                    this.drawScaledLetter(ctx, aa, x, yOffset, charWidth, drawHeight, colorStr, clipRect);
                    
                    ctx.restore(); // Restore from clipping
                }
                
                // Update yOffset for next letter (move upward, no gap)
                yOffset -= drawHeight;
            }
        }
        
        // Draw selection indicator bars at the top if there are selections
        if (this.selectionRanges && this.selectionRanges.length > 0) {
            ctx.fillStyle = 'rgba(251, 191, 36, 0.8)';
            
            for (const range of this.selectionRanges) {
                const startX = logoX + range.start * charWidth;
                const endX = logoX + (range.end + 1) * charWidth;
                const width = endX - startX;
                ctx.fillRect(startX, logoY - 8, width, 4);
            }
            
            // Draw selection range text for first range only to avoid clutter
            const firstRange = this.selectionRanges[0];
            const startX = logoX + firstRange.start * charWidth;
            ctx.fillStyle = '#92400E';
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'left';
            const rangeText = this.selectionRanges.length === 1 
                ? `${firstRange.start + 1}-${firstRange.end + 1}`
                : `${this.selectionRanges.length} selections`;
            ctx.fillText(rangeText, startX, logoY - 10);
        }
        
        // Draw query sequence
        if (this.parsedSeqs.length > 0) {
            const querySeq = this.parsedSeqs[0].sequence;
            ctx.font = '10px monospace';
            ctx.textBaseline = 'middle';
            
            for (let pos = 0; pos < querySeq.length; pos++) {
                const aa = querySeq[pos].toUpperCase();
                const x = logoX + pos * charWidth;
                const isSelected = this.isPositionSelected(pos);
                
                // Use Dayhoff colors (same as py2Dmol Logo mode)
                let color = this.getDayhoffColor(aa);
                
                // Darken if selected
                if (isSelected) {
                    color = {
                        r: Math.round(color.r * 0.5),
                        g: Math.round(color.g * 0.5),
                        b: Math.round(color.b * 0.5)
                    };
                }
                
                ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                ctx.fillRect(x, queryY, charWidth, queryRowHeight);
                
                ctx.fillStyle = isSelected ? '#fff' : '#000';
                ctx.textAlign = 'center';
                ctx.fillText(aa, x + charWidth / 2, queryY + queryRowHeight / 2);
            }
        }
        
        // Draw black bar above query
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(logoX, queryY);
        ctx.lineTo(logoX + this.maxSeqLen * charWidth, queryY);
        ctx.stroke();
        
        // Draw tick marks (position numbers)
        ctx.fillStyle = '#666';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        for (let i = 0; i < this.maxSeqLen; i += 10) {
            const x = logoX + i * charWidth + charWidth / 2;
            ctx.fillText(String(i + 1), x, tickY + 8);
        }
    }
}
'''


SEQUENCE_VIEWER_JS = '''
class SequenceViewer {
    constructor(container, sequences, probabilities, structureViewer = null, msaViewer = null) {
        this.container = container;
        this.sequences = sequences;
        this.probabilities = probabilities;
        this.structureViewer = structureViewer;
        this.msaViewer = msaViewer;
        this.colorMode = 'probability';  // 'probability', 'charge', 'hydrophobicity', 'conservation', 'none'
        this.selectedSeqIndex = 0;
        
        // Multi-range selection state
        this.selectionRanges = [];  // Array of { start, end } objects
        this.currentSelectionStart = null;
        this.currentSelectionEnd = null;
        this.isSelecting = false;
        
        // Charge classification
        this.positiveCharge = new Set(['R', 'K', 'H']);
        this.negativeCharge = new Set(['D', 'E']);
        
        // Hydrophobicity scale (Kyte-Doolittle)
        this.hydrophobicity = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
            'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
            'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        };
        
        // Pre-compute conservation scores
        this.conservationScores = this.computeConservation();
        
        this.render();
    }
    
    setStructureViewer(viewer) {
        this.structureViewer = viewer;
    }
    
    setMSAViewer(viewer) {
        this.msaViewer = viewer;
    }
    
    // Get all selection ranges (array of { start, end } objects)
    getSelectionRanges() {
        return this.selectionRanges.length > 0 ? this.selectionRanges : null;
    }
    
    // Get current in-progress selection range (normalized so start <= end)
    getCurrentSelectionRange() {
        if (this.currentSelectionStart === null || this.currentSelectionEnd === null) {
            return null;
        }
        const start = Math.min(this.currentSelectionStart, this.currentSelectionEnd);
        const end = Math.max(this.currentSelectionStart, this.currentSelectionEnd);
        return { start, end };
    }
    
    // Check if a residue index is in any selection range
    isResidueInSelection(index) {
        // Check current in-progress selection
        const currentRange = this.getCurrentSelectionRange();
        if (currentRange && index >= currentRange.start && index <= currentRange.end) {
            return true;
        }
        // Check finalized selections
        for (const range of this.selectionRanges) {
            if (index >= range.start && index <= range.end) {
                return true;
            }
        }
        return false;
    }
    
    // Add a new selection range (merge overlapping ranges)
    addSelectionRange(start, end) {
        const newRange = { start: Math.min(start, end), end: Math.max(start, end) };
        
        // Merge with existing ranges if overlapping or adjacent
        const mergedRanges = [];
        let merged = false;
        
        for (const existing of this.selectionRanges) {
            // Check if ranges overlap or are adjacent
            if (newRange.start <= existing.end + 1 && newRange.end >= existing.start - 1) {
                // Merge ranges
                newRange.start = Math.min(newRange.start, existing.start);
                newRange.end = Math.max(newRange.end, existing.end);
                merged = true;
            } else {
                mergedRanges.push(existing);
            }
        }
        
        mergedRanges.push(newRange);
        
        // Sort by start position
        mergedRanges.sort((a, b) => a.start - b.start);
        
        // Second pass to merge any newly adjacent ranges
        this.selectionRanges = [];
        for (const range of mergedRanges) {
            if (this.selectionRanges.length === 0) {
                this.selectionRanges.push(range);
            } else {
                const last = this.selectionRanges[this.selectionRanges.length - 1];
                if (range.start <= last.end + 1) {
                    last.end = Math.max(last.end, range.end);
                } else {
                    this.selectionRanges.push(range);
                }
            }
        }
        
        this.syncSelectionWithViewers();
        this.updateSelectionHighlight();
    }
    
    // Toggle selection of a single residue (for Ctrl+click)
    toggleResidueSelection(index) {
        // Check if residue is already selected
        let foundInRange = -1;
        for (let i = 0; i < this.selectionRanges.length; i++) {
            const range = this.selectionRanges[i];
            if (index >= range.start && index <= range.end) {
                foundInRange = i;
                break;
            }
        }
        
        if (foundInRange >= 0) {
            // Remove from selection - may need to split range
            const range = this.selectionRanges[foundInRange];
            this.selectionRanges.splice(foundInRange, 1);
            
            if (index > range.start) {
                this.selectionRanges.push({ start: range.start, end: index - 1 });
            }
            if (index < range.end) {
                this.selectionRanges.push({ start: index + 1, end: range.end });
            }
            
            // Sort by start position
            this.selectionRanges.sort((a, b) => a.start - b.start);
        } else {
            // Add to selection
            this.addSelectionRange(index, index);
            return; // addSelectionRange already syncs
        }
        
        this.syncSelectionWithViewers();
        this.updateSelectionHighlight();
    }
    
    // Sync selection with other viewers
    syncSelectionWithViewers() {
        const ranges = this.getSelectionRanges();
        
        if (this.structureViewer) {
            this.structureViewer.setSelectionRanges(ranges);
        }
        
        if (this.msaViewer) {
            this.msaViewer.setSelectionRanges(ranges);
        }
    }
    
    // Set selection during drag (updates current selection)
    setCurrentSelection(start, end) {
        this.currentSelectionStart = start;
        this.currentSelectionEnd = end;
        
        // Create temporary combined ranges for display
        const tempRanges = [...this.selectionRanges];
        const currentRange = this.getCurrentSelectionRange();
        if (currentRange) {
            tempRanges.push(currentRange);
        }
        
        // Sync with viewers using combined ranges
        if (this.structureViewer) {
            this.structureViewer.setSelectionRanges(tempRanges.length > 0 ? tempRanges : null);
        }
        if (this.msaViewer) {
            this.msaViewer.setSelectionRanges(tempRanges.length > 0 ? tempRanges : null);
        }
        
        this.updateSelectionHighlight();
    }
    
    // Finalize current selection
    finalizeSelection() {
        const currentRange = this.getCurrentSelectionRange();
        if (currentRange) {
            this.addSelectionRange(currentRange.start, currentRange.end);
        }
        this.currentSelectionStart = null;
        this.currentSelectionEnd = null;
    }
    
    // Clear all selections
    clearSelection() {
        this.selectionRanges = [];
        this.currentSelectionStart = null;
        this.currentSelectionEnd = null;
        this.isSelecting = false;
        
        if (this.structureViewer) {
            this.structureViewer.setSelectionRanges(null);
        }
        
        if (this.msaViewer) {
            this.msaViewer.setSelectionRanges(null);
        }
        
        this.updateSelectionHighlight();
    }
    
    // Get total count of selected residues
    getSelectedCount() {
        let count = 0;
        for (const range of this.selectionRanges) {
            count += range.end - range.start + 1;
        }
        return count;
    }
    
    // Format selection ranges as string
    formatSelectionRanges() {
        if (this.selectionRanges.length === 0) return '';
        
        const parts = this.selectionRanges.map(r => {
            if (r.start === r.end) {
                return `${r.start + 1}`;
            }
            return `${r.start + 1}-${r.end + 1}`;
        });
        
        return parts.join(', ');
    }
    
    // Update visual selection highlight on residues
    updateSelectionHighlight() {
        this.container.querySelectorAll('.residue[data-residue-index]').forEach(residueEl => {
            const index = parseInt(residueEl.dataset.residueIndex);
            
            if (this.isResidueInSelection(index)) {
                residueEl.classList.add('selected');
            } else {
                residueEl.classList.remove('selected');
            }
        });
        
        // Update selection info display
        const selectionInfo = this.container.querySelector('.selection-info');
        if (selectionInfo) {
            const hasSelection = this.selectionRanges.length > 0 || this.getCurrentSelectionRange() !== null;
            if (hasSelection) {
                const count = this.getSelectedCount();
                const currentRange = this.getCurrentSelectionRange();
                // Include current selection in count if present
                const currentCount = currentRange ? (currentRange.end - currentRange.start + 1) : 0;
                const totalCount = count + currentCount;
                const rangeText = this.formatSelectionRanges();
                const currentText = currentRange ? 
                    (rangeText ? `, ${currentRange.start + 1}-${currentRange.end + 1}` : `${currentRange.start + 1}-${currentRange.end + 1}`) : '';
                selectionInfo.textContent = `Selected: ${rangeText}${currentText} (${totalCount} residues)`;
                selectionInfo.style.display = 'inline-flex';
            } else {
                selectionInfo.style.display = 'none';
            }
        }
        
        // Show/hide clear button
        const clearBtn = this.container.querySelector('.clear-selection-btn');
        if (clearBtn) {
            const hasSelection = this.selectionRanges.length > 0 || this.getCurrentSelectionRange() !== null;
            clearBtn.style.display = hasSelection ? 'inline-block' : 'none';
        }
    }
    
    computeConservation() {
        // Compute conservation score at each position based on frequency of most common AA
        const conservation = [];
        if (this.sequences.length === 0) return conservation;
        
        const seqLen = this.sequences[0].sequence.replace(/:/g, '').length;
        
        for (let pos = 0; pos < seqLen; pos++) {
            const aaCounts = {};
            let total = 0;
            
            for (const seq of this.sequences) {
                const cleanSeq = seq.sequence.replace(/:/g, '');
                if (pos < cleanSeq.length) {
                    const aa = cleanSeq[pos].toUpperCase();
                    if (aa !== '-' && aa !== 'X') {
                        aaCounts[aa] = (aaCounts[aa] || 0) + 1;
                        total++;
                    }
                }
            }
            
            // Conservation = frequency of most common AA
            let maxFreq = 0;
            for (const count of Object.values(aaCounts)) {
                const freq = count / total;
                if (freq > maxFreq) maxFreq = freq;
            }
            conservation.push(maxFreq);
        }
        
        return conservation;
    }
    
    getChargeColor(aa) {
        if (this.positiveCharge.has(aa)) {
            return '#3B82F6';  // Blue - positive
        } else if (this.negativeCharge.has(aa)) {
            return '#EF4444';  // Red - negative
        }
        return '#E5E7EB';  // Gray - neutral
    }
    
    getHydrophobicityColor(aa) {
        const value = this.hydrophobicity[aa.toUpperCase()];
        if (value === undefined) return '#E5E7EB';
        
        // Map from -4.5 to 4.5 to color gradient
        // Hydrophobic (positive) = Orange/Brown
        // Hydrophilic (negative) = Blue
        const normalized = (value + 4.5) / 9.0;  // 0 to 1
        
        if (normalized > 0.5) {
            // Hydrophobic - orange to brown
            const intensity = (normalized - 0.5) * 2;
            const r = Math.round(234 - intensity * 50);
            const g = Math.round(179 - intensity * 100);
            const b = Math.round(8 + intensity * 20);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Hydrophilic - light blue to blue
            const intensity = (0.5 - normalized) * 2;
            const r = Math.round(147 - intensity * 88);
            const g = Math.round(197 - intensity * 60);
            const b = Math.round(253 - intensity * 9);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }
    
    getConservationColor(position) {
        const score = this.conservationScores[position];
        if (score === undefined) return '#E5E7EB';
        
        // High conservation = purple, low = light gray
        if (score >= 0.9) {
            return '#7C3AED';  // Purple - highly conserved
        } else if (score >= 0.7) {
            return '#A78BFA';  // Light purple
        } else if (score >= 0.5) {
            return '#C4B5FD';  // Very light purple
        } else {
            return '#E5E7EB';  // Gray - variable
        }
    }
    
    getResidueColor(residueIndex, seqIndex) {
        const seq = this.sequences[seqIndex];
        if (!seq) return null;
        
        const cleanSeq = seq.sequence.replace(/:/g, '');
        const aa = cleanSeq[residueIndex]?.toUpperCase();
        if (!aa) return null;
        
        switch (this.colorMode) {
            case 'charge':
                return this.getChargeColor(aa);
            case 'hydrophobicity':
                return this.getHydrophobicityColor(aa);
            case 'conservation':
                return this.getConservationColor(residueIndex);
            case 'probability':
                return this.getProbabilityColor(residueIndex, seqIndex);
            case 'none':
            default:
                return null;
        }
    }
    
    // Check if a sequence has probability data
    hasProbabilityData(seqIndex) {
        if (!this.probabilities) return false;
        
        const seq = this.sequences[seqIndex];
        if (!seq) return false;
        
        const headerParts = seq.header.split(',');
        const seqName = headerParts[0].trim();
        
        const probData = this.probabilities[seqName];
        return !!(probData && probData.residue_probabilities);
    }
    
    getProbabilityColor(residueIndex, seqIndex) {
        if (!this.probabilities) return null;
        
        const seq = this.sequences[seqIndex];
        if (!seq) return null;
        
        const headerParts = seq.header.split(',');
        const seqName = headerParts[0].trim();
        
        const probData = this.probabilities[seqName];
        if (!probData || !probData.residue_probabilities) return null;
        
        const residueKey = `residue_${residueIndex + 1}`;
        const residueProbs = probData.residue_probabilities[residueKey];
        if (!residueProbs) return null;
        
        let maxProb = 0;
        for (const [aa, prob] of Object.entries(residueProbs)) {
            if (aa !== 'X' && prob > maxProb) {
                maxProb = prob;
            }
        }
        
        // AlphaFold pLDDT color scale:
        if (maxProb <= 0.5) {
            return '#FF7E45';  // Orange - low confidence
        } else if (maxProb <= 0.7) {
            return '#FFDB12';  // Yellow - medium confidence
        } else if (maxProb <= 0.9) {
            return '#57CAF9';  // Light Blue - high confidence
        } else {
            return '#0053D7';  // Dark Blue - very high confidence
        }
    }
    
    getMaxProbability(residueIndex, seqIndex) {
        if (!this.probabilities) return null;
        
        const seq = this.sequences[seqIndex];
        if (!seq) return null;
        
        const headerParts = seq.header.split(',');
        const seqName = headerParts[0].trim();
        
        const probData = this.probabilities[seqName];
        if (!probData || !probData.residue_probabilities) return null;
        
        const residueKey = `residue_${residueIndex + 1}`;
        const residueProbs = probData.residue_probabilities[residueKey];
        if (!residueProbs) return null;
        
        let maxProb = 0;
        for (const [aa, prob] of Object.entries(residueProbs)) {
            if (aa !== 'X' && prob > maxProb) {
                maxProb = prob;
            }
        }
        
        return maxProb;
    }
    
    getAllProbabilityScores(seqIndex = 0) {
        // Get probability scores for all residues as an array
        const seq = this.sequences[seqIndex];
        if (!seq || !this.probabilities) return null;
        
        const headerParts = seq.header.split(',');
        const seqName = headerParts[0].trim();
        
        const probData = this.probabilities[seqName];
        if (!probData || !probData.residue_probabilities) return null;
        
        const cleanSeq = seq.sequence.replace(/:/g, '');
        const scores = [];
        
        for (let i = 0; i < cleanSeq.length; i++) {
            const residueKey = `residue_${i + 1}`;
            const residueProbs = probData.residue_probabilities[residueKey];
            
            if (!residueProbs) {
                scores.push(null);
                continue;
            }
            
            let maxProb = 0;
            for (const [aa, prob] of Object.entries(residueProbs)) {
                if (aa !== 'X' && prob > maxProb) {
                    maxProb = prob;
                }
            }
            scores.push(maxProb);
        }
        
        return scores;
    }
    
    setColorMode(mode) {
        this.colorMode = mode;
        
        // Sync with structure viewer if available
        if (this.structureViewer) {
            // Map sequence viewer modes to structure viewer modes
            const structureMode = (mode === 'none') ? 'chain' : mode;
            this.structureViewer.setColorMode(structureMode);
        }
        
        this.render();
    }
    
    selectSequence(index) {
        this.selectedSeqIndex = index;
        
        // Update structure viewer when sequence changes
        if (this.structureViewer) {
            // Update probability scores
            const probScores = this.getAllProbabilityScores(index);
            this.structureViewer.setProbabilityScores(probScores);
            
            // Update sequence for charge/hydrophobicity coloring
            const seq = this.sequences[index];
            if (seq) {
                this.structureViewer.setSequence(seq.sequence);
            }
        }
        
        this.render();
    }
    
    parseHeader(header) {
        const parts = header.split(',').map(p => p.trim());
        const name = parts[0];
        const meta = {};
        
        for (let i = 1; i < parts.length; i++) {
            const [key, value] = parts[i].split(':');
            if (key && value) {
                meta[key.trim()] = value.trim();
            }
        }
        
        return { name, meta };
    }
    
    render() {
        const seq = this.sequences[this.selectedSeqIndex];
        if (!seq) return;
        
        const { name, meta } = this.parseHeader(seq.header);
        const isReference = this.selectedSeqIndex === 0;
        
        // Build tabs with reference/designed distinction
        let tabsHtml = '<div class="sequence-selector-container">';
        
        // Reference sequence section
        if (this.sequences.length > 0) {
            const { name: refName } = this.parseHeader(this.sequences[0].header);
            const refActiveClass = this.selectedSeqIndex === 0 ? 'active' : '';
            tabsHtml += `
                <div class="sequence-group">
                    <span class="sequence-group-label reference-label">Reference</span>
                    <div class="sequence-selector">
                        <button class="sequence-tab reference-tab ${refActiveClass}" data-index="0">${refName}</button>
                    </div>
                </div>
            `;
        }
        
        // Designed sequences section
        if (this.sequences.length > 1) {
            tabsHtml += `
                <div class="sequence-group">
                    <span class="sequence-group-label designed-label">Designed (${this.sequences.length - 1})</span>
                    <div class="sequence-selector">
            `;
            for (let i = 1; i < this.sequences.length; i++) {
                const { name: sName } = this.parseHeader(this.sequences[i].header);
                const activeClass = i === this.selectedSeqIndex ? 'active' : '';
                tabsHtml += `<button class="sequence-tab designed-tab ${activeClass}" data-index="${i}">${sName}</button>`;
            }
            tabsHtml += `
                    </div>
                </div>
            `;
        }
        
        tabsHtml += '</div>';
        
        let metaHtml = '';
        if (Object.keys(meta).length > 0) {
            metaHtml = '<div class="metadata-grid" style="margin-bottom: 12px;">';
            for (const [key, value] of Object.entries(meta)) {
                metaHtml += `
                    <div class="metadata-item">
                        <div class="metadata-label">${key.replace(/_/g, ' ')}</div>
                        <div class="metadata-value">${value}</div>
                    </div>`;
            }
            metaHtml += '</div>';
        }
        
        const sequence = seq.sequence;
        const chains = sequence.split(':');
        let seqHtml = '<div class="sequence-content">';
        
        let globalIndex = 0;
        chains.forEach((chain, chainIdx) => {
            if (chainIdx > 0) {
                seqHtml += '<span style="color: var(--color-gray-400); margin: 0 4px; font-weight: bold;">:</span>';
            }
            
            for (let i = 0; i < chain.length; i++) {
                const residue = chain[i];
                const color = this.getResidueColor(globalIndex, this.selectedSeqIndex);
                
                if (color && this.colorMode !== 'none') {
                    const textColor = this.colorMode === 'hydrophobicity' || this.colorMode === 'charge' ? '#000' : '#fff';
                    seqHtml += `<span class="residue" data-residue-index="${globalIndex}" data-residue-name="${residue}" style="background:${color}; color:${textColor}; font-weight:500;">${residue}</span>`;
                } else {
                    seqHtml += `<span class="residue" data-residue-index="${globalIndex}" data-residue-name="${residue}">${residue}</span>`;
                }
                globalIndex++;
            }
        });
        seqHtml += '</div>';
        
        // Build legend based on color mode
        let legendHtml = '';
        const hasProbData = this.hasProbabilityData(this.selectedSeqIndex);
        
        if (this.colorMode === 'probability') {
            if (hasProbData) {
                legendHtml = `
                    <div class="color-legend">
                        <div class="legend-blocks">
                            <div class="legend-block">
                                <div class="legend-color" style="background:#FF7E45;"></div>
                                <span>≤0.5</span>
                            </div>
                            <div class="legend-block">
                                <div class="legend-color" style="background:#FFDB12;"></div>
                                <span>0.5-0.7</span>
                            </div>
                            <div class="legend-block">
                                <div class="legend-color" style="background:#57CAF9;"></div>
                                <span>0.7-0.9</span>
                            </div>
                            <div class="legend-block">
                                <div class="legend-color" style="background:#0053D7;"></div>
                                <span>>0.9</span>
                            </div>
                        </div>
                    </div>`;
            } else {
                legendHtml = `
                    <div class="color-legend" style="color: var(--color-gray-500); font-style: italic;">
                        No probability data available for this sequence
                    </div>`;
            }
        } else if (this.colorMode === 'charge') {
            legendHtml = `
                <div class="color-legend">
                    <div class="legend-blocks">
                        <div class="legend-block">
                            <div class="legend-color" style="background:#3B82F6;"></div>
                            <span>+ (R,K,H)</span>
                        </div>
                        <div class="legend-block">
                            <div class="legend-color" style="background:#EF4444;"></div>
                            <span>− (D,E)</span>
                        </div>
                        <div class="legend-block">
                            <div class="legend-color" style="background:#E5E7EB;"></div>
                            <span>Neutral</span>
                        </div>
                    </div>
                </div>`;
        } else if (this.colorMode === 'hydrophobicity') {
            legendHtml = `
                <div class="color-legend">
                    <div class="legend-blocks">
                        <div class="legend-block">
                            <div class="legend-color" style="background:#3B82F6;"></div>
                            <span>Hydrophilic</span>
                        </div>
                        <div class="legend-block">
                            <div class="legend-color" style="background:#93C5FD;"></div>
                            <span>Polar</span>
                        </div>
                        <div class="legend-block">
                            <div class="legend-color" style="background:#EAB308;"></div>
                            <span>Hydrophobic</span>
                        </div>
                    </div>
                </div>`;
        } else if (this.colorMode === 'conservation') {
            legendHtml = `
                <div class="color-legend">
                    <div class="legend-blocks">
                        <div class="legend-block">
                            <div class="legend-color" style="background:#7C3AED;"></div>
                            <span>≥90%</span>
                        </div>
                        <div class="legend-block">
                            <div class="legend-color" style="background:#A78BFA;"></div>
                            <span>70-90%</span>
                        </div>
                        <div class="legend-block">
                            <div class="legend-color" style="background:#C4B5FD;"></div>
                            <span>50-70%</span>
                        </div>
                        <div class="legend-block">
                            <div class="legend-color" style="background:#E5E7EB;"></div>
                            <span><50%</span>
                        </div>
                    </div>
                </div>`;
        }
        
        const typeBadge = isReference 
            ? '<span class="sequence-type-badge reference-badge">Reference</span>'
            : '<span class="sequence-type-badge designed-badge">Designed</span>';
        
        const hasSelection = this.selectionRanges.length > 0;
        const selectionDisplay = hasSelection ? 'inline-flex' : 'none';
        const clearBtnDisplay = hasSelection ? 'inline-block' : 'none';
        const selectionText = hasSelection ? `Selected: ${this.formatSelectionRanges()} (${this.getSelectedCount()} residues)` : '';
        
        this.container.innerHTML = `
            ${tabsHtml}
            <div class="sequence-entry ${isReference ? 'reference-entry' : 'designed-entry'}">
                <div class="sequence-header">
                    <div class="sequence-name-container">
                        <span class="sequence-name">${name}</span>
                        ${typeBadge}
                    </div>
                    <span class="sequence-meta">${sequence.replace(/:/g, '').length} residues</span>
                </div>
                ${metaHtml}
                <div class="color-mode-controls">
                    <button class="btn btn-xs ${this.colorMode === 'probability' ? 'btn-primary' : 'btn-secondary'}" data-color-mode="probability">Probability</button>
                    <button class="btn btn-xs ${this.colorMode === 'charge' ? 'btn-primary' : 'btn-secondary'}" data-color-mode="charge">Charge</button>
                    <button class="btn btn-xs ${this.colorMode === 'hydrophobicity' ? 'btn-primary' : 'btn-secondary'}" data-color-mode="hydrophobicity">Hydrophobicity</button>
                    <button class="btn btn-xs ${this.colorMode === 'conservation' ? 'btn-primary' : 'btn-secondary'}" data-color-mode="conservation">Conservation</button>
                    <button class="btn btn-xs ${this.colorMode === 'none' ? 'btn-primary' : 'btn-secondary'}" data-color-mode="none">None</button>
                    <span class="selection-info" style="display: ${selectionDisplay}; margin-left: 10px; background: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 500; align-items: center; gap: 4px;">
                        ${selectionText}
                    </span>
                    <button class="btn btn-xs btn-secondary clear-selection-btn" style="display: ${clearBtnDisplay}; margin-left: 4px;">Clear Selection</button>
                </div>
                ${legendHtml}
                <div class="sequence-selection-hint" style="font-size: 0.7rem; color: var(--color-gray-500); margin-bottom: 6px; flex-shrink: 0;">
                    <em>Click and drag to select residues. Hold Ctrl/Cmd to add multiple selections.</em>
                </div>
                ${seqHtml}
            </div>
        `;
        
        this.container.querySelectorAll('.sequence-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                this.selectSequence(index);
            });
        });
        
        // Color mode button listeners
        this.container.querySelectorAll('[data-color-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setColorMode(e.target.dataset.colorMode);
            });
        });
        
        // Clear selection button listener
        const clearBtn = this.container.querySelector('.clear-selection-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearSelection();
            });
        }
        
        // Add selection and hover event listeners for residues
        const sequenceContent = this.container.querySelector('.sequence-content');
        if (sequenceContent) {
            // Prevent text selection during drag
            sequenceContent.style.userSelect = 'none';
            sequenceContent.style.webkitUserSelect = 'none';
            sequenceContent.style.cursor = 'pointer';
        }
        
        this.container.querySelectorAll('.residue[data-residue-index]').forEach(residueEl => {
            // Mouse down - start selection
            residueEl.addEventListener('mousedown', (e) => {
                e.preventDefault();
                const residueIndex = parseInt(e.target.dataset.residueIndex);
                const isAdditive = e.ctrlKey || e.metaKey;  // Ctrl on Windows/Linux, Cmd on Mac
                
                this.isSelecting = true;
                this.isAdditiveSelection = isAdditive;
                
                // If not additive, clear existing selections
                if (!isAdditive) {
                    this.selectionRanges = [];
                }
                
                this.currentSelectionStart = residueIndex;
                this.currentSelectionEnd = residueIndex;
                this.setCurrentSelection(residueIndex, residueIndex);
            });
            
            // Mouse enter while selecting - extend selection
            residueEl.addEventListener('mouseenter', (e) => {
                const residueIndex = parseInt(e.target.dataset.residueIndex);
                const residueName = e.target.dataset.residueName;
                
                if (this.isSelecting) {
                    this.currentSelectionEnd = residueIndex;
                    this.setCurrentSelection(this.currentSelectionStart, this.currentSelectionEnd);
                } else {
                    // Normal hover behavior when not selecting
                    const probability = this.getMaxProbability(residueIndex, this.selectedSeqIndex);
                    if (this.structureViewer) {
                        this.structureViewer.highlightResidue(residueIndex, residueName, probability);
                    }
                }
            });
            
            residueEl.addEventListener('mouseleave', () => {
                if (!this.isSelecting && this.structureViewer) {
                    this.structureViewer.clearHighlight();
                }
            });
        });
        
        // Mouse up - end selection (listen on document to catch mouseup outside)
        const mouseUpHandler = () => {
            if (this.isSelecting) {
                this.isSelecting = false;
                // Finalize selection - add current range to selection ranges
                this.finalizeSelection();
            }
        };
        
        document.addEventListener('mouseup', mouseUpHandler);
        
        // Apply selection highlight if there's an existing selection
        this.updateSelectionHighlight();
    }
}
'''


APP_JS = '''
document.addEventListener('DOMContentLoaded', function() {
    const contentDiv = document.getElementById('content');
    
    // Create summary header showing included files
    const prefixes = Object.keys(triflowData.groups);
    const summaryHeader = document.createElement('div');
    summaryHeader.className = 'summary-header';
    
    let totalSequences = 0;
    let totalStructures = 0;
    prefixes.forEach(prefix => {
        totalSequences += triflowData.groups[prefix].sequences.length;
        totalStructures += triflowData.groups[prefix].structures.length;
    });
    
    summaryHeader.innerHTML = `
        <div class="summary-info">
            <span class="summary-stat"><strong>${totalSequences}</strong> sequence${totalSequences !== 1 ? 's' : ''}</span>
            <span class="summary-divider">•</span>
            <span class="summary-stat"><strong>${totalStructures}</strong> structure${totalStructures !== 1 ? 's' : ''}</span>
        </div>
        <div class="summary-files">
            <span class="summary-files-label">Included files:</span>
            ${prefixes.map(p => `<span class="summary-file-tag" data-prefix="${p}">${p}</span>`).join('')}
        </div>
    `;
    contentDiv.appendChild(summaryHeader);
    
    // Add click handlers to file tags for scrolling to that group
    summaryHeader.querySelectorAll('.summary-file-tag').forEach(tag => {
        tag.style.cursor = 'pointer';
        tag.addEventListener('click', () => {
            const prefix = tag.dataset.prefix;
            const groupEl = document.getElementById(`group-${prefix}`);
            if (groupEl) {
                groupEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
    
    for (const [prefix, groupData] of Object.entries(triflowData.groups)) {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'group-container';
        groupDiv.id = `group-${prefix}`;
        
        const headerHtml = `
            <div class="group-header">
                <span class="group-title">${prefix}</span>
                <span class="group-badge">${groupData.sequences.length} sequences</span>
            </div>
        `;
        
        let contentHtml = '<div class="content-grid">';
        
        // Structure viewer panel
        contentHtml += `
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">3D Structure</span>
                </div>
                <div class="panel-body">
                    <div class="structure-viewer-container">
                        <canvas class="structure-canvas" id="canvas-${prefix}"></canvas>
                    </div>
                    <div class="viewer-controls">
                        <div class="control-group">
                            <label class="control-label">Color:</label>
                            <select id="colorSelect-${prefix}">
                                <option value="rainbow">Rainbow</option>
                                <option value="chain" selected>Chain</option>
                                <option value="secondary">Secondary Structure</option>
                                <option value="probability">Probability</option>
                                <option value="bfactor">B-factor</option>
                                <option value="charge">Charge</option>
                                <option value="hydrophobicity">Hydrophobicity</option>
                                <option value="conservation">Conservation</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <label class="control-label">Outline:</label>
                            <select id="outlineSelect-${prefix}">
                                <option value="full">Full</option>
                                <option value="partial">Partial</option>
                                <option value="none">None</option>
                            </select>
                        </div>
                        <div class="toggle-wrapper">
                            <span class="control-label">Shadow</span>
                            <div class="toggle active" id="shadowToggle-${prefix}"></div>
                        </div>
                        <div class="toggle-wrapper">
                            <span class="control-label">Depth</span>
                            <div class="toggle" id="depthToggle-${prefix}"></div>
                        </div>
                    </div>
                    <div class="viewer-controls">
                        <div class="slider-control">
                            <label class="control-label">Width:</label>
                            <input type="range" id="widthSlider-${prefix}" min="1" max="8" step="0.5" value="3">
                            <span class="slider-value" id="widthValue-${prefix}">3.0</span>
                        </div>
                        <button class="btn btn-secondary btn-sm" id="rotateBtn-${prefix}">Auto Rotate</button>
                        <button class="btn btn-secondary btn-sm" id="resetBtn-${prefix}">Reset</button>
                        <button class="btn btn-primary btn-sm" id="saveSvgBtn-${prefix}">Save SVG</button>
                        <button class="btn btn-primary btn-sm" id="savePngBtn-${prefix}">Save PNG</button>
                        <span class="viewer-info" id="viewerInfo-${prefix}"></span>
                    </div>
                </div>
            </div>
        `;
        
        // Individual sequence viewer panel
        contentHtml += `
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Single Sequence View</span>
                </div>
                <div class="panel-body">
                    <div class="sequence-viewer" id="seqViewer-${prefix}"></div>
                </div>
            </div>
        `;
        
        contentHtml += '</div>';
        
        // MSA viewer panel (full width)
        contentHtml += `
            <div class="panel panel-full" style="margin-top: 20px;">
                <div class="panel-header">
                    <span class="panel-title">Sequence Profile</span>
                </div>
                <div class="panel-body">
                    <div class="msa-container" id="msaViewer-${prefix}"></div>
                </div>
            </div>
        `;
        
        
        groupDiv.innerHTML = headerHtml + contentHtml;
        contentDiv.appendChild(groupDiv);
        
        // Initialize viewers after DOM is ready
        setTimeout(() => {
            initializeViewers(prefix, groupData);
        }, 50);
    }
});

function initializeViewers(prefix, groupData) {
    let structureViewer = null;
    
    // Initialize structure viewer
    const canvas = document.getElementById(`canvas-${prefix}`);
    if (canvas && groupData.structures.length > 0) {
        // Set canvas dimensions
        const container = canvas.parentElement;
        const rect = container.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        canvas.width = (rect.width || 500) * dpr;
        canvas.height = 450 * dpr;
        canvas.style.width = (rect.width || 500) + 'px';
        canvas.style.height = '450px';
        
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        
        const pdbContent = groupData.structures[0].pdb;
        structureViewer = new StructureViewer(canvas, pdbContent, {
            shadow: true,
            outline: 'full',
            width: 3.0,
            depth: false,
            secondaryStructure: groupData.secondaryStructure || []
        });
        
        // Update info
        const infoEl = document.getElementById(`viewerInfo-${prefix}`);
        if (infoEl) {
            infoEl.textContent = `${structureViewer.getAtomCount()} residues, ${structureViewer.getChainCount()} chain(s)`;
        }
        
        // Setup controls
        const colorSelect = document.getElementById(`colorSelect-${prefix}`);
        const outlineSelect = document.getElementById(`outlineSelect-${prefix}`);
        const shadowToggle = document.getElementById(`shadowToggle-${prefix}`);
        const depthToggle = document.getElementById(`depthToggle-${prefix}`);
        const widthSlider = document.getElementById(`widthSlider-${prefix}`);
        const widthValue = document.getElementById(`widthValue-${prefix}`);
        const rotateBtn = document.getElementById(`rotateBtn-${prefix}`);
        const resetBtn = document.getElementById(`resetBtn-${prefix}`);
        const saveSvgBtn = document.getElementById(`saveSvgBtn-${prefix}`);
        const savePngBtn = document.getElementById(`savePngBtn-${prefix}`);
        
        if (colorSelect) {
            colorSelect.addEventListener('change', (e) => {
                structureViewer.setColorMode(e.target.value);
            });
        }
        
        if (outlineSelect) {
            outlineSelect.addEventListener('change', (e) => {
                structureViewer.setOutline(e.target.value);
            });
        }
        
        if (shadowToggle) {
            shadowToggle.addEventListener('click', () => {
                shadowToggle.classList.toggle('active');
                structureViewer.setShadow(shadowToggle.classList.contains('active'));
            });
        }
        
        if (depthToggle) {
            depthToggle.addEventListener('click', () => {
                depthToggle.classList.toggle('active');
                structureViewer.setDepth(depthToggle.classList.contains('active'));
            });
        }
        
        if (widthSlider) {
            widthSlider.addEventListener('input', (e) => {
                const width = parseFloat(e.target.value);
                if (widthValue) widthValue.textContent = width.toFixed(1);
                structureViewer.setLineWidth(width);
            });
        }
        
        if (rotateBtn) {
            rotateBtn.addEventListener('click', () => {
                const isRotating = structureViewer.toggleAutoRotate();
                rotateBtn.classList.toggle('active', isRotating);
            });
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                structureViewer.resetView();
                if (rotateBtn) rotateBtn.classList.remove('active');
            });
        }
        
        if (saveSvgBtn) {
            saveSvgBtn.addEventListener('click', () => {
                structureViewer.downloadSVG(`${prefix}_structure.svg`);
            });
        }
        
        if (savePngBtn) {
            savePngBtn.addEventListener('click', () => {
                structureViewer.downloadPNG(`${prefix}_structure.png`);
            });
        }
    }
    
    // Initialize sequence viewer with reference to structure viewer
    let sequenceViewer = null;
    const seqContainer = document.getElementById(`seqViewer-${prefix}`);
    if (seqContainer && groupData.sequences.length > 0) {
        sequenceViewer = new SequenceViewer(seqContainer, groupData.sequences, groupData.probabilities, structureViewer);
        
        // Pass conservation, probability scores, and sequence to structure viewer
        if (structureViewer) {
            if (sequenceViewer.conservationScores) {
                structureViewer.setConservationScores(sequenceViewer.conservationScores);
            }
            
            // Set initial sequence for charge/hydrophobicity coloring
            if (groupData.sequences[0]) {
                structureViewer.setSequence(groupData.sequences[0].sequence);
            }
            
            // Set probability scores for the reference sequence (index 0)
            // This will be null if reference doesn't have probability data, which is correct
            const probScores = sequenceViewer.getAllProbabilityScores(0);
            structureViewer.setProbabilityScores(probScores);
        }
    }
    
    // Initialize MSA viewer
    let msaViewer = null;
    const msaContainer = document.getElementById(`msaViewer-${prefix}`);
    if (msaContainer && groupData.sequences.length > 0) {
        msaViewer = new MSAViewer(msaContainer, groupData.sequences);
    }
    
    // Connect sequence viewer to MSA viewer for selection sync
    if (sequenceViewer && msaViewer) {
        sequenceViewer.setMSAViewer(msaViewer);
    }
}
'''


def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML summary from TriFlow output directory'
    )
    parser.add_argument(
        'output_dir',
        help='Path to TriFlow output directory'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output HTML file path (default: output_dir/summary.html)'
    )
    
    args = parser.parse_args()
    
    generate_html(args.output_dir, args.output)


if __name__ == '__main__':
    main()
