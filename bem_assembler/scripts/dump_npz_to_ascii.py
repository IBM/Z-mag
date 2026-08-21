#!/usr/bin/env python3
"""
Dump NPZ data files to ASCII format for verification.
This allows comparing with C++ binary loader output.
"""

import numpy as np
import os
import argparse


def dump_bnd_data(npz_path, txt_path):
    """Dump bnd_data_rankN.npz to ASCII."""
    print(f"Dumping {npz_path} -> {txt_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    with open(txt_path, 'w') as f:
        # n_bnd_nodes
        n_bnd_nodes = int(data['n_bnd_nodes'])
        f.write(f"{n_bnd_nodes}\n")
        
        # ghost_bnd_nodes_indices
        ghost_indices = data['ghost_bnd_nodes_indices'].tolist()
        f.write(f"{len(ghost_indices)}\n")
        for owner, local_idx in ghost_indices:
            f.write(f"{owner} {local_idx}\n")
        
        # bnd_nodes_coordinates
        coords = data['bnd_nodes_coordinates']
        f.write(f"{len(coords)}\n")
        for coord in coords:
            f.write(f"{coord[0]:.16e} {coord[1]:.16e} {coord[2]:.16e}\n")
        
        # bnd_f_to_bnd_n
        facets = data['bnd_f_to_bnd_n']
        f.write(f"{len(facets)}\n")
        for facet in facets:
            f.write(f"{facet[0]} {facet[1]} {facet[2]}\n")
        
        # vertex_to_dof
        v2d = data['vertex_to_dof']
        f.write(f"{len(v2d)}\n")
        for val in v2d:
            f.write(f"{val}\n")
    
    print(f"  ✓ Dumped successfully")


def dump_bnd_nodes(npz_path, txt_path):
    """Dump bnd_nodes_rankN.npz to ASCII."""
    print(f"Dumping {npz_path} -> {txt_path}")
    
    data = np.load(npz_path)
    
    with open(txt_path, 'w') as f:
        nodes = data['bnd_nodes']
        f.write(f"{len(nodes)}\n")
        for val in nodes:
            f.write(f"{val}\n")
    
    print(f"  ✓ Dumped successfully")


def dump_mesh_topology(npz_path, txt_path):
    """Dump mesh_topology_rankN.npz to ASCII."""
    print(f"Dumping {npz_path} -> {txt_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    with open(txt_path, 'w') as f:
        # node_to_cell
        n2c = data['node_to_cell']
        f.write(f"{len(n2c)}\n")
        for cells in n2c:
            f.write(f"{len(cells)}")
            for cell in cells:
                f.write(f" {cell}")
            f.write("\n")
        
        # cell_to_node
        c2n = data['cell_to_node']
        f.write(f"{len(c2n)}\n")
        for nodes in c2n:
            f.write(f"{len(nodes)}")
            for node in nodes:
                f.write(f" {node}")
            f.write("\n")
        
        # topology_nodes_coordinates
        coords = data['topology_nodes_coordinates']
        f.write(f"{len(coords)}\n")
        for coord in coords:
            f.write(f"{coord[0]:.16e} {coord[1]:.16e} {coord[2]:.16e}\n")
        
        # topology_bnd_nodes
        bnd_nodes = data['topology_bnd_nodes']
        f.write(f"{len(bnd_nodes)}\n")
        for val in bnd_nodes:
            f.write(f"{val}\n")
    
    print(f"  ✓ Dumped successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Dump NPZ data files to ASCII for verification'
    )
    parser.add_argument('--rank', type=int, default=0, help='Rank to dump')
    parser.add_argument('--data-dir', default='bem_py/standalone/data_to_load',
                       help='Input directory with NPZ files')
    parser.add_argument('--output-dir', default='bem_py/standalone/data_dump_python',
                       help='Output directory for ASCII files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NPZ to ASCII Dumper")
    print("=" * 60)
    
    # Check input directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Input directory '{args.data_dir}' not found!")
        sys.exit(1)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    rank = args.rank
    
    # Dump bnd_data
    npz_path = os.path.join(args.data_dir, f'bnd_data_rank{rank}.npz')
    txt_path = os.path.join(args.output_dir, f'bnd_data_rank{rank}.txt')
    if os.path.exists(npz_path):
        dump_bnd_data(npz_path, txt_path)
    else:
        print(f"  ⚠ Warning: {npz_path} not found")
    
    # Dump bnd_nodes
    npz_path = os.path.join(args.data_dir, f'bnd_nodes_rank{rank}.npz')
    txt_path = os.path.join(args.output_dir, f'bnd_nodes_rank{rank}.txt')
    if os.path.exists(npz_path):
        dump_bnd_nodes(npz_path, txt_path)
    else:
        print(f"  ⚠ Warning: {npz_path} not found")
    
    # Dump mesh_topology
    npz_path = os.path.join(args.data_dir, f'mesh_topology_rank{rank}.npz')
    txt_path = os.path.join(args.output_dir, f'mesh_topology_rank{rank}.txt')
    if os.path.exists(npz_path):
        dump_mesh_topology(npz_path, txt_path)
    else:
        print(f"  ⚠ Warning: {npz_path} not found")
    
    print("\n" + "=" * 60)
    print(f"✓ Dump complete for rank {rank}!")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    main()

