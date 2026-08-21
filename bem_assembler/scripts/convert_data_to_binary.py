#!/usr/bin/env python3
"""
Convert NPZ data files to binary format for C++ consumption.

This script converts:
- bnd_data_rankN.npz -> bnd_data_rankN.bin
- bnd_nodes_rankN.npz -> bnd_nodes_rankN.bin  
- mesh_topology_rankN.npz -> mesh_topology_rankN.bin

Usage:
    python convert_data_to_binary.py [--rank N] [--size N]
"""

import numpy as np
import struct
import os
import argparse
import sys


def write_vector_int(f, vec):
    """Write a vector of ints: [length, data...]"""
    f.write(struct.pack('q', len(vec)))  # int64
    for val in vec:
        f.write(struct.pack('i', int(val)))  # int32


def write_vector_double(f, vec):
    """Write a vector of doubles: [length, data...]"""
    f.write(struct.pack('q', len(vec)))  # int64
    for val in vec:
        f.write(struct.pack('d', float(val)))  # double


def write_vector3d(f, vec):
    """Write a 3D vector [x, y, z]"""
    f.write(struct.pack('ddd', float(vec[0]), float(vec[1]), float(vec[2])))


def write_matrix_double(f, mat):
    """Write a 2D matrix: [rows, cols, data...]"""
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    f.write(struct.pack('qq', rows, cols))  # int64 x2
    for row in mat:
        for val in row:
            f.write(struct.pack('d', float(val)))


def convert_bnd_data(npz_path, bin_path):
    """Convert bnd_data_rankN.npz to binary format."""
    print(f"Converting {npz_path} -> {bin_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    with open(bin_path, 'wb') as f:
        # Write header
        f.write(b'BND_DATA')  # 8-byte magic number
        f.write(struct.pack('i', 1))  # version
        
        # n_bnd_nodes
        n_bnd_nodes = int(data['n_bnd_nodes'])
        f.write(struct.pack('i', n_bnd_nodes))
        
        # ghost_bnd_nodes_indices: list of (owner, local_index) pairs
        ghost_indices = data['ghost_bnd_nodes_indices'].tolist()
        f.write(struct.pack('q', len(ghost_indices)))
        for owner, local_idx in ghost_indices:
            f.write(struct.pack('ii', int(owner), int(local_idx)))
        
        # bnd_nodes_coordinates: Nx3 array
        coords = data['bnd_nodes_coordinates']
        f.write(struct.pack('q', len(coords)))
        for coord in coords:
            write_vector3d(f, coord)
        
        # bnd_f_to_bnd_n: Nx3 array (facet to node mapping)
        facets = data['bnd_f_to_bnd_n']
        f.write(struct.pack('q', len(facets)))
        for facet in facets:
            f.write(struct.pack('iii', int(facet[0]), int(facet[1]), int(facet[2])))
        
        # vertex_to_dof: 1D array
        v2d = data['vertex_to_dof']
        write_vector_int(f, v2d)
    
    print(f"  ✓ Converted successfully")


def convert_bnd_nodes(npz_path, bin_path):
    """Convert bnd_nodes_rankN.npz to binary format."""
    print(f"Converting {npz_path} -> {bin_path}")
    
    data = np.load(npz_path)
    
    with open(bin_path, 'wb') as f:
        # Write header
        f.write(b'BND_NODE')  # 8-byte magic number
        f.write(struct.pack('i', 1))  # version
        
        # bnd_nodes: 1D array of node indices
        nodes = data['bnd_nodes']
        write_vector_int(f, nodes)
    
    print(f"  ✓ Converted successfully")


def convert_mesh_topology(npz_path, bin_path):
    """Convert mesh_topology_rankN.npz to binary format."""
    print(f"Converting {npz_path} -> {bin_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    with open(bin_path, 'wb') as f:
        # Write header
        f.write(b'MESH_TOP')  # 8-byte magic number
        f.write(struct.pack('i', 1))  # version
        
        # node_to_cell: list of lists
        n2c = data['node_to_cell']
        f.write(struct.pack('q', len(n2c)))
        for cells in n2c:
            write_vector_int(f, cells)
        
        # cell_to_node: list of lists
        c2n = data['cell_to_node']
        f.write(struct.pack('q', len(c2n)))
        for nodes in c2n:
            write_vector_int(f, nodes)
        
        # topology_nodes_coordinates: Nx3 array
        coords = data['topology_nodes_coordinates']
        f.write(struct.pack('q', len(coords)))
        for coord in coords:
            write_vector3d(f, coord)
        
        # topology_bnd_nodes: 1D array
        bnd_nodes = data['topology_bnd_nodes']
        write_vector_int(f, bnd_nodes)
    
    print(f"  ✓ Converted successfully")


def convert_rank(rank, data_dir, output_dir):
    """Convert all data files for a given rank."""
    print(f"\n=== Converting data for rank {rank} ===")
    
    # Convert bnd_data
    npz_path = os.path.join(data_dir, f'bnd_data_rank{rank}.npz')
    bin_path = os.path.join(output_dir, f'bnd_data_rank{rank}.bin')
    if os.path.exists(npz_path):
        convert_bnd_data(npz_path, bin_path)
    else:
        print(f"  ⚠ Warning: {npz_path} not found, skipping")
    
    # Convert bnd_nodes
    npz_path = os.path.join(data_dir, f'bnd_nodes_rank{rank}.npz')
    bin_path = os.path.join(output_dir, f'bnd_nodes_rank{rank}.bin')
    if os.path.exists(npz_path):
        convert_bnd_nodes(npz_path, bin_path)
    else:
        print(f"  ⚠ Warning: {npz_path} not found, skipping")
    
    # Convert mesh_topology
    npz_path = os.path.join(data_dir, f'mesh_topology_rank{rank}.npz')
    bin_path = os.path.join(output_dir, f'mesh_topology_rank{rank}.bin')
    if os.path.exists(npz_path):
        convert_mesh_topology(npz_path, bin_path)
    else:
        print(f"  ⚠ Warning: {npz_path} not found, skipping")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPZ data files to binary format for C++',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all ranks (auto-detect)
  python convert_data_to_binary.py

  # Convert specific rank
  python convert_data_to_binary.py --rank 0 --size 4

  # Convert all ranks with known size
  python convert_data_to_binary.py --size 80
        """
    )
    parser.add_argument('--rank', type=int, help='Specific rank to convert')
    parser.add_argument('--size', type=int, help='Total number of ranks')
    parser.add_argument('--data-dir', default='bem_py/standalone/data_to_load',
                       help='Input directory with NPZ files')
    parser.add_argument('--output-dir', default='bem_py/standalone/data_binary',
                       help='Output directory for binary files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NPZ to Binary Converter for C++")
    print("=" * 60)
    
    # Check input directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Input directory '{args.data_dir}' not found!")
        print(f"Please run from the root of the bem directory.")
        sys.exit(1)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Determine which ranks to convert
    if args.rank is not None:
        # Convert specific rank
        ranks = [args.rank]
    elif args.size is not None:
        # Convert all ranks up to size
        ranks = range(args.size)
    else:
        # Auto-detect by looking for files
        ranks = []
        for f in os.listdir(args.data_dir):
            if f.startswith('bnd_data_rank') and f.endswith('.npz'):
                rank_num = int(f.replace('bnd_data_rank', '').replace('.npz', ''))
                ranks.append(rank_num)
        ranks = sorted(set(ranks))
        
        if not ranks:
            print(f"Error: No bnd_data_rank*.npz files found in {args.data_dir}")
            sys.exit(1)
        
        print(f"Auto-detected {len(ranks)} ranks: {ranks[:5]}{'...' if len(ranks) > 5 else ''}")
    
    # Convert each rank
    for rank in ranks:
        try:
            convert_rank(rank, args.data_dir, args.output_dir)
        except Exception as e:
            print(f"  ✗ Error converting rank {rank}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ Conversion complete!")
    print(f"  Input:  {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Ranks:  {len(ranks)}")
    print("=" * 60)
    
    print("\nBinary file format:")
    print("  bnd_data_rankN.bin   - Boundary data for rank N")
    print("  bnd_nodes_rankN.bin  - Boundary nodes for rank N")
    print("  mesh_topology_rankN.bin - Mesh topology for rank N")
    
    print("\nTo use in C++:")
    print("  1. Implement binary readers in your C++ code")
    print("  2. See convert_data_to_binary.py for file format details")
    print("  3. Or use the Python version which has full NPZ support!")


if __name__ == "__main__":
    main()

