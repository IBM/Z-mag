#include <iostream>
#include <chrono>
#include <sys/stat.h>
#include "bem_assembler.hpp"

// Helper function to check if directory exists
bool directory_exists(const std::string& path) {
    struct stat info;
    return (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));
}

// Helper function to create directory
void create_directory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

// Simple argument parser
struct Args {
    int rank = 0;
    int size = 1;
    
    static Args parse(int argc, char** argv) {
        Args args;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--rank" && i + 1 < argc) {
                args.rank = std::atoi(argv[++i]);
            } else if (arg == "--size" && i + 1 < argc) {
                args.size = std::atoi(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [OPTIONS]\n";
                std::cout << "Options:\n";
                std::cout << "  --rank N    Local rank (default: 0)\n";
                std::cout << "  --size N    World size (default: 1)\n";
                std::cout << "  --help,-h   Show this help\n";
                std::exit(0);
            }
        }
        return args;
    }
};

int main(int argc, char** argv) {
    // Parse command line arguments
    Args args = Args::parse(argc, argv);
    int rank = args.rank;
    int size = args.size;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  BEM Matrix Assembly (CPU-only)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[Rank " << rank << "] Running with " << size << " virtual ranks" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Working directories
    std::string base_dir = "./standalone";
    std::string data_dir = base_dir + "/data_to_load";
    std::string ref_dir = base_dir + "/BEM_computed";
    std::string output_dir = base_dir + "/BEM_computed_cpp";
    std::string dump_dir = base_dir + "/data_dump_cpp";
    
    // Create output directories
    if (rank == 0) {
        if (!directory_exists(output_dir)) {
            create_directory(output_dir);
        }
        if (!directory_exists(dump_dir)) {
            create_directory(dump_dir);
        }
    }
    
    // ========================================
    // STEP 1: Load Data
    // ========================================
    
    bem::DataLoader loader;
    
    // Load boundary data for all ranks
    std::vector<bem::BoundaryData> bnd_data_per_rank;
    std::vector<int> global_n_bnd_nodes;
    
    std::cout << "[Rank " << rank << "] Loading boundary data..." << std::endl;
    
    if (!loader.load_boundary_data_ascii(rank, size, data_dir, bnd_data_per_rank, global_n_bnd_nodes)) {
        std::cerr << "\n[Rank " << rank << "] ERROR: Failed to load boundary data." << std::endl;
        std::cerr << "[Rank " << rank << "] Please ensure ASCII files exist in: " << data_dir << std::endl;
        std::cerr << "[Rank " << rank << "] Run convert_data_to_ascii.py to convert NPZ files first." << std::endl;
        return 1;
    }
    
    // Load mesh topology
    bem::MeshTopology mesh_topology;
    if (!loader.load_mesh_topology_ascii(rank, data_dir, mesh_topology)) {
        std::cerr << "\n[Rank " << rank << "] ERROR: Failed to load mesh topology." << std::endl;
        return 1;
    }
    
    // Load boundary nodes
    std::vector<int> bnd_nodes;
    if (!loader.load_boundary_nodes_ascii(rank, data_dir, bnd_nodes)) {
        std::cerr << "\n[Rank " << rank << "] ERROR: Failed to load boundary nodes." << std::endl;
        return 1;
    }
    
    // ========================================
    // STEP 2: Dump Data for Verification
    // ========================================
    
    std::cout << "[Rank " << rank << "] Dumping loaded data to ASCII for verification..." << std::endl;
    
    // Dump boundary data for this rank
    std::string dump_file = dump_dir + "/bnd_data_rank" + std::to_string(rank) + ".txt";
    loader.dump_bnd_data_ascii(dump_file, bnd_data_per_rank[rank], global_n_bnd_nodes[rank]);
    
    // Dump boundary nodes
    dump_file = dump_dir + "/bnd_nodes_rank" + std::to_string(rank) + ".txt";
    loader.dump_bnd_nodes_ascii(dump_file, bnd_nodes);
    
    // Dump mesh topology
    dump_file = dump_dir + "/mesh_topology_rank" + std::to_string(rank) + ".txt";
    loader.dump_mesh_topology_ascii(dump_file, mesh_topology);
    
    std::cout << "[Rank " << rank << "] ASCII dumps written to: " << dump_dir << std::endl;
    
    // ========================================
    // STEP 3: Assemble BEM Matrix
    // ========================================
    
    std::cout << "\n[Rank " << rank << "] Assembling BEM matrix..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bem::BEMAssembler assembler;
    bem::Matrix B_local = assembler.assemble(
        rank, size,
        bnd_data_per_rank,
        global_n_bnd_nodes,
        bnd_nodes,
        mesh_topology
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_time = duration.count() / 1000.0;
    
    std::cout << "\n[Rank " << rank << "] Time for BEM matrix assembly: " 
              << total_time << " seconds" << std::endl;
    
    // ========================================
    // STEP 4: Save Results
    // ========================================
    
    // Save binary version
    std::string output_filename_bin = output_dir + "/B_matrix_slice_rank" + std::to_string(rank) + ".bin";
    assembler.save_matrix(output_filename_bin, B_local);
    
    // Save ASCII version
    std::string output_filename_txt = output_dir + "/B_matrix_slice_rank" + std::to_string(rank) + ".txt";
    assembler.save_matrix_ascii(output_filename_txt, B_local);
    
    // ========================================
    // STEP 5: Compare with Reference
    // ========================================
    
    std::string ref_filename = ref_dir + "/B_matrix_slice_rank" + std::to_string(rank) + ".txt";
    if (std::ifstream(ref_filename)) {
        std::cout << "\n[Rank " << rank << "] Comparing with reference matrix..." << std::endl;
        
        bem::Matrix B_ref = assembler.load_matrix_ascii(ref_filename);
        
        if (B_ref.size() > 0) {
            double max_diff, avg_diff;
            size_t num_mismatches;
            double rtol = 1e-6;
            double atol = 1e-8;
            
            bool passed = assembler.compare_matrices(B_local, B_ref, rtol, atol, max_diff, avg_diff, num_mismatches);
            
            std::cout << "\n[Rank " << rank << "] Comparison results:" << std::endl;
            std::cout << "  Matrix size: " << B_local.size() << " x " 
                      << (B_local.size() > 0 ? B_local[0].size() : 0) << std::endl;
            std::cout << "  Max difference: " << max_diff << std::endl;
            std::cout << "  Avg difference: " << avg_diff << std::endl;
            std::cout << "  Mismatches: " << num_mismatches << " / " 
                      << (B_local.size() * (B_local.size() > 0 ? B_local[0].size() : 0)) << std::endl;
            std::cout << "  Tolerance: rtol=" << rtol << ", atol=" << atol << std::endl;
            
            if (passed) {
                std::cout << "  ✓ PASSED: Matrices match within tolerance" << std::endl;
            } else {
                std::cout << "  ✗ FAILED: Matrices differ beyond tolerance" << std::endl;
            }
        }
    } else {
        std::cout << "\n[Rank " << rank << "] No reference matrix found at: " << ref_filename << std::endl;
        std::cout << "[Rank " << rank << "] Skipping comparison. To compare:" << std::endl;
        std::cout << "[Rank " << rank << "]   1. Run Python version to generate reference" << std::endl;
        std::cout << "[Rank " << rank << "]   2. Convert with: python scripts/convert_matrix_to_ascii.py --size " << size << std::endl;
        std::cout << "[Rank " << rank << "]   3. Re-run this program" << std::endl;
    }
    
    // ========================================
    // Done!
    // ========================================
    
    std::cout << "\n[Rank " << rank << "] BEM matrix assembly completed successfully!" << std::endl;
    std::cout << "[Rank " << rank << "] Results saved to: " << output_dir << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
