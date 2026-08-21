"""
Main script demonstrating BEM matrix assembly using the unified BEMAssembler class.

This script shows how to:
1. Load a mesh with boundary markers
2. Instantiate the BEMAssembler
3. Assemble the BEM matrix
4. Save results in format compatible with validation scripts

Usage:
    python main.py
    
or with MPI:
    mpirun -n 80 python main.py
    
Validation:
    # Compare full global matrices (recommended - works with any decomposition)
    python scripts/compare_global_matrices.py ./output ./standalone/BEM_computed_cpp_debug
    
    # Or compare per-rank slices (requires same MPI decomposition)
    python scripts/validate_results.py ./output ./standalone/BEM_computed_cpp_debug
"""

import numpy as np
import sys
import time
from mpi4py import MPI
from dolfinx.io import XDMFFile

# Import the BEM assembler
from bem_assembler import BEMAssembler


def load_mesh_and_markers(mesh_file, comm):
    """
    Load mesh and facet tags from XDMF file.
    
    Parameters
    ----------
    mesh_file : str
        Path to mesh XDMF file (without extension)
    comm : MPI.Comm
        MPI communicator
    
    Returns
    -------
    tuple
        (domain, facet_tags)
    """
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"Loading mesh from: {mesh_file}.xdmf")
    
    with XDMFFile(comm, f"{mesh_file}.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
        
        # Create topology entities (required for facet tags)
        # This fixes "Missing IndexMap in Topology" error
        tdim = domain.topology.dim
        domain.topology.create_entities(tdim - 1)  # Create facets
        
        facet_tags = xdmf.read_meshtags(domain, name="facet_tags")
    
    if rank == 0:
        print(f"  ✓ Mesh loaded")
        print(f"  ✓ Dimension: {domain.topology.dim}")
        print(f"  ✓ Num cells (local): {domain.topology.index_map(domain.topology.dim).size_local}")
        print(f"  ✓ Num facets (local): {domain.topology.index_map(domain.topology.dim-1).size_local}")
    
    return domain, facet_tags


def main():
    """Main BEM assembly workflow."""
    
    # ===================================================================
    # MPI Setup
    # ===================================================================
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # ===================================================================
    # Configuration
    # ===================================================================
    # Mesh file path (adjust to your mesh location)
    # Expected files: mesh.xdmf, mesh.h5
    # mesh_file = "./standalone/data_to_load/domain"
    mesh_file = "./test_mesh/domain"
    
    # Boundary surface marker (marker identifying the magnetic material boundary)
    # Adjust this value based on your mesh tags
    surface_marker = 10  # Change to match your mesh's boundary marker
    
    # Output directory (format matches validate_results.py expectations)
    output_dir = "./output"
    output_file = f"{output_dir}/B_matrix_slice_rank{rank}.npy"
    
    # Print startup message
    if rank == 0:
        print("\n" + "="*70)
        print("  BEM Matrix Assembly - Main Script")
        print("="*70)
        print(f"MPI ranks: {size}")
        print(f"Mesh file: {mesh_file}.xdmf")
        print("="*70 + "\n")
    
    # ===================================================================
    # Step 1: Load Mesh
    # ===================================================================
    # Check if mesh files exist
    import os
    if rank == 0:
        if not os.path.exists(f"{mesh_file}.xdmf"):
            print(f"\nERROR: Mesh file not found: {mesh_file}.xdmf")
            sys.exit(1)
    
    comm.Barrier()  # Wait for rank 0 to check
    
    try:
        domain, facet_tags = load_mesh_and_markers(mesh_file, comm)
    except Exception as e:
        if rank == 0:
            print(f"\nERROR: Could not load mesh from {mesh_file}.xdmf")
            print(f"Error: {e}")
            print("\nPlease ensure:")
            print("  1. Mesh file exists at the specified path")
            print("  2. Mesh has facet tags named 'facet_tags'")
            print("  3. Adjust 'mesh_file' path in main.py")
        sys.exit(1)
    
    # ===================================================================
    # Step 2: Create BEM Assembler
    # ===================================================================
    if rank == 0:
        print("\nCreating BEM assembler...")
    
    assembler = BEMAssembler(
        domain=domain,
        facet_tags=facet_tags,
        surface_marker=surface_marker,
        verbose=True     # Print progress (rank 0 only)
    )
    
    # ===================================================================
    # Step 3: Assemble BEM Matrix
    # ===================================================================
    if rank == 0:
        print("\nStarting BEM assembly...")
    
    start_time = time.time()
    
    # This single call does all the work:
    # - Computes boundary node mapping
    # - Gathers data from all ranks
    # - Assembles BEM matrix (off-diagonal + solid angle diagonal)
    B_local = assembler.assemble()
    
    end_time = time.time()
    assembly_time = end_time - start_time
    

    # Gather timing statistics
    all_times = comm.gather(assembly_time, root=0)
    
    # Print summary statistics (rank 0 only)
    if rank == 0:
        print(f"\n{'='*70}")
        print("  Assembly Statistics")
        print(f"{'='*70}")
        print(f"Local matrix shape: {B_local.shape}")
        print(f"Assembly time: {assembly_time:.4f} seconds")
        if all_times:
            avg_time = np.mean(all_times)
            min_time = np.min(all_times)
            max_time = np.max(all_times)
            print(f"Time across all ranks: avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")
        print(f"{'='*70}\n")
    
    comm.Barrier()
    
    if rank == 0:
        avg_time = np.mean(all_times)
        max_time = np.max(all_times)
        min_time = np.min(all_times)
        print(f"\n{'='*70}")
        print("  Timing Statistics (all ranks)")
        print(f"{'='*70}")
        print(f"  Average: {avg_time:.4f} seconds")
        print(f"  Min:     {min_time:.4f} seconds")
        print(f"  Max:     {max_time:.4f} seconds")
        print(f"{'='*70}")
        
        # Print matrix dimensions
        total_local_nodes = sum(assembler.global_n_bnd_nodes)
        print(f"\n{'='*70}")
        print("  Matrix Distribution")
        print(f"{'='*70}")
        print(f"  Total boundary nodes: {total_local_nodes}")
        print(f"  Nodes per rank: {assembler.global_n_bnd_nodes}")
        print(f"  Global matrix size: ({total_local_nodes}, {total_local_nodes})")
        print(f"{'='*70}\n")
    
    # ===================================================================
    # Step 5: Save Results
    # ===================================================================
    if rank == 0:
        print("Saving BEM matrix...")
    
    # Save using assembler's save method
    assembler.save(output_file)
    
    # Also save as text for inspection
    output_file_txt = output_file.replace('.npy', '.txt')
    # Directory already created by assembler.save() above
    with open(output_file_txt, 'w') as f:
        rows, cols = B_local.shape
        f.write(f"{rows} {cols}\n")
        if rows > 0 and cols > 0:
            np.savetxt(f, B_local, fmt='%.16e')
    
    if rank == 0:
        print(f"  ✓ Binary: {output_file}")
        print(f"  ✓ Text:   {output_file_txt}")
    
    # ===================================================================
    # Optional: Test Load
    # ===================================================================
    if rank == 0:
        print("\nTesting load functionality...")
    
    # Create a new assembler and load the saved matrix
    assembler2 = BEMAssembler(domain, facet_tags, surface_marker, verbose=False)
    B_loaded, _, _, _, _ = assembler2.load(output_file)
    
    # Verify loaded matrix matches
    if np.allclose(B_loaded, B_local):
        if rank == 0:
            print("  ✓ Load test passed: Loaded matrix matches assembled matrix")
    else:
        print(f"[Rank {rank}] WARNING: Loaded matrix differs from assembled matrix!")
    
    # ===================================================================
    # Completion
    # ===================================================================
    if rank == 0:
        print(f"\n{'='*70}")
        print("  BEM Assembly Complete!")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}/")
        print(f"Output files: B_matrix_slice_rank*.npy, B_matrix_slice_rank*.txt")
        print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

