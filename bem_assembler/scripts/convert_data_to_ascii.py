#!/usr/bin/env python3
"""
Convert NPZ data files to ASCII format for C++ consumption.

This script converts all NPZ files in a directory to ASCII format:
- bnd_data_rankN.npz -> bnd_data_rankN.txt
- bnd_nodes_rankN.npz -> bnd_nodes_rankN.txt  
- mesh_topology_rankN.npz -> mesh_topology_rankN.txt

Usage:
    python convert_data_to_ascii.py [--input-dir DIR] [--output-dir DIR] [--size N]
"""

import numpy as np
import os
import argparse
import sys


def convert_bnd_data(npz_path, txt_path):
    """Convert bnd_data_rankN.npz to ASCII."""
    print(f"Converting {npz_path} -> {txt_path}")
    
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
    
    print(f"  ✓ Converted successfully")


def convert_bnd_nodes(npz_path, txt_path):
    """Convert bnd_nodes_rankN.npz to ASCII."""
    print(f"Converting {npz_path} -> {txt_path}")
    
    data = np.load(npz_path)
    
    with open(txt_path, 'w') as f:
        nodes = data['bnd_nodes']
        f.write(f"{len(nodes)}\n")
        for val in nodes:
            f.write(f"{val}\n")
    
    print(f"  ✓ Converted successfully")


def convert_mesh_topology(npz_path, txt_path):
    """Convert mesh_topology_rankN.npz to ASCII."""
    print(f"Converting {npz_path} -> {txt_path}")
    
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
    
    print(f"  ✓ Converted successfully")


def convert_rank(rank, input_dir, output_dir):
    """Convert all data files for a given rank."""
    print(f"\n=== Converting data for rank {rank} ===")
    
    # Convert bnd_data
    npz_path = os.path.join(input_dir, f'bnd_data_rank{rank}.npz')
    txt_path = os.path.join(output_dir, f'bnd_data_rank{rank}.txt')
    if os.path.exists(npz_path):
        try:
            convert_bnd_data(npz_path, txt_path)
        except Exception as e:
            print(f"  ✗ Error converting bnd_data: {e}")
    else:
        print(f"  ⚠ Warning: {npz_path} not found, skipping")
    
    # Convert bnd_nodes
    npz_path = os.path.join(input_dir, f'bnd_nodes_rank{rank}.npz')
    txt_path = os.path.join(output_dir, f'bnd_nodes_rank{rank}.txt')
    if os.path.exists(npz_path):
        try:
            convert_bnd_nodes(npz_path, txt_path)
        except Exception as e:
            print(f"  ✗ Error converting bnd_nodes: {e}")
    else:
        print(f"  ⚠ Warning: {npz_path} not found, skipping")
    
    # Convert mesh_topology
    npz_path = os.path.join(input_dir, f'mesh_topology_rank{rank}.npz')
    txt_path = os.path.join(output_dir, f'mesh_topology_rank{rank}.txt')
    if os.path.exists(npz_path):
        try:
            convert_mesh_topology(npz_path, txt_path)
        except Exception as e:
            print(f"  ✗ Error converting mesh_topology: {e}")
    else:
        print(f"  ⚠ Warning: {npz_path} not found, skipping")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPZ data files to ASCII format for C++',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all ranks (auto-detect) in default directory
  python convert_data_to_ascii.py

  # Convert all ranks with known size
  python convert_data_to_ascii.py --size 80

  # Convert to different output directory
  python convert_data_to_ascii.py --output-dir ./ascii_data

  # Convert from custom input directory
  python convert_data_to_ascii.py --input-dir ./custom_npz --output-dir ./custom_ascii
        """
    )
    parser.add_argument('--input-dir', default='bem_py/standalone/data_to_load',
                       help='Input directory with NPZ files (default: bem_py/standalone/data_to_load)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for ASCII files (default: same as input-dir)')
    parser.add_argument('--size', type=int, help='Total number of ranks to convert')
    
    args = parser.parse_args()
    
    # Default output dir to input dir if not specified
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print("=" * 60)
    print("NPZ to ASCII Converter")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Check input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found!")
        sys.exit(1)
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Determine which ranks to convert
    if args.size is not None:
        # Convert all ranks up to size
        ranks = range(args.size)
        print(f"Converting ranks 0 to {args.size - 1}")
    else:
        # Auto-detect by looking for files
        ranks = []
        for f in os.listdir(args.input_dir):
            if f.startswith('bnd_data_rank') and f.endswith('.npz'):
                rank_num = int(f.replace('bnd_data_rank', '').replace('.npz', ''))
                ranks.append(rank_num)
        ranks = sorted(set(ranks))
        
        if not ranks:
            print(f"Error: No bnd_data_rank*.npz files found in {args.input_dir}")
            sys.exit(1)
        
        print(f"Auto-detected {len(ranks)} ranks: {ranks[:10]}{'...' if len(ranks) > 10 else ''}")
    
    # Convert each rank
    success_count = 0
    error_count = 0
    
    for rank in ranks:
        try:
            convert_rank(rank, args.input_dir, args.output_dir)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error converting rank {rank}: {e}")
            error_count += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Conversion complete!")
    print(f"  Successfully converted: {success_count} ranks")
    if error_count > 0:
        print(f"  Failed: {error_count} ranks")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)
    
    print("\nASCII file format:")
    print("  bnd_data_rankN.txt       - Boundary data for rank N")
    print("  bnd_nodes_rankN.txt      - Boundary nodes for rank N")
    print("  mesh_topology_rankN.txt  - Mesh topology for rank N")
    
    print("\nTo use in C++:")
    print("  The C++ program will now load directly from ASCII files!")
    print(f"  Make sure your data_dir points to: {args.output_dir}")


if __name__ == "__main__":
    main()

