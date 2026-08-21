#pragma once

#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include "utils.hpp"

namespace bem {

using Matrix = std::vector<std::vector<double>>;

/**
 * BEM matrix assembler (Header-only implementation)
 */
class BEMAssembler {
public:
    BEMAssembler() = default;
    
    /**
     * HIGH-PERFORMANCE ZERO-COPY API
     * Direct pointer + size interface for maximum performance in memory-critical workflows
     * 
     * This interface accepts raw pointers to contiguous arrays, avoiding all intermediate copies.
     * Arrays must be in C-contiguous (row-major) layout.
     * 
     * Per-rank data layout:
     * - coords_data: [rank0_node0_x, rank0_node0_y, rank0_node0_z, rank0_node1_x, ...]
     * - facets_data: [rank0_facet0_n0, rank0_facet0_n1, rank0_facet0_n2, rank0_facet1_n0, ...]
     * - ghost_indices_data: [rank0_ghost0_owner, rank0_ghost0_idx, rank0_ghost1_owner, ...]
     * 
     * @param rank: Current MPI rank
     * @param size: Total number of MPI ranks
     * @param coords_ptrs: Array of pointers to coordinate arrays (one per rank), shape [size][n_nodes*3]
     * @param n_nodes_per_rank: Number of nodes per rank, shape [size]
     * @param facets_ptrs: Array of pointers to facet arrays (one per rank), shape [size][n_facets*3]
     * @param n_facets_per_rank: Number of facets per rank, shape [size]
     * @param ghost_indices_ptrs: Array of pointers to ghost index arrays, shape [size][n_ghosts*2]
     * @param n_ghosts_per_rank: Number of ghost nodes per rank, shape [size]
     * @param global_n_bnd_nodes: Number of boundary nodes per rank, shape [size]
     * @param output_buffer: Pre-allocated output buffer [n_local x n_global], or nullptr to allocate
     * @param solid_angles_ptr: Optional solid angles for diagonal correction
     * @param solid_angles_size: Size of solid angles array
     * 
     * @return Local matrix. If output_buffer is provided, values are written in-place
     *         and an empty matrix is returned.
     * 
     * This is a low-level API. Bounds are validated and will raise on invalid input.
     */
    inline Matrix assemble_direct(
        int rank,
        int size,
        const double* const* coords_ptrs,          // [size][n_nodes*3]
        const int* n_nodes_per_rank,               // [size]
        const int* const* facets_ptrs,             // [size][n_facets*3]
        const int* n_facets_per_rank,              // [size]
        const int* const* ghost_indices_ptrs,      // [size][n_ghosts*2]
        const int* n_ghosts_per_rank,              // [size]
        const int* global_n_bnd_nodes,             // [size]
        double* output_buffer = nullptr,           // [n_local*n_global] or nullptr
        const double* solid_angles_ptr = nullptr,
        size_t solid_angles_size = 0
    ) {
        int n_local = global_n_bnd_nodes[rank];
        int n_global = 0;
        for (int i = 0; i < size; ++i) {
            n_global += global_n_bnd_nodes[i];
        }
        
        
        const bool use_output_buffer = (output_buffer != nullptr);
        Matrix B;
        if (!use_output_buffer) {
            B.resize(n_local, std::vector<double>(n_global, 0.0));
        }

        auto add_entry = [&](int i, int j, double v) {
            if (i < 0 || i >= n_local || j < 0 || j >= n_global) {
                throw std::runtime_error(
                    "BEM add_entry index out of bounds: i=" + std::to_string(i) +
                    ", j=" + std::to_string(j) +
                    ", n_local=" + std::to_string(n_local) +
                    ", n_global=" + std::to_string(n_global)
                );
            }
            if (use_output_buffer) {
                output_buffer[static_cast<size_t>(i) * static_cast<size_t>(n_global) + static_cast<size_t>(j)] += v;
            } else {
                B[i][j] += v;
            }
        };
        auto set_entry = [&](int i, int j, double v) {
            if (i < 0 || i >= n_local || j < 0 || j >= n_global) {
                throw std::runtime_error(
                    "BEM set_entry index out of bounds: i=" + std::to_string(i) +
                    ", j=" + std::to_string(j) +
                    ", n_local=" + std::to_string(n_local) +
                    ", n_global=" + std::to_string(n_global)
                );
            }
            if (use_output_buffer) {
                output_buffer[static_cast<size_t>(i) * static_cast<size_t>(n_global) + static_cast<size_t>(j)] = v;
            } else {
                B[i][j] = v;
            }
        };
        
        // Off-diagonal: loop over all ranks' boundary facets; fill owned rows of this rank.
        for (int k = 0; k < size; ++k) {
            const double* foreign_coords = coords_ptrs[k];      // [n_nodes*3]
            const int* foreign_facets = facets_ptrs[k];         // [n_facets*3]
            const int* foreign_ghosts = ghost_indices_ptrs[k];  // [n_ghosts*2]
            const int n_nodes_k = n_nodes_per_rank[k];
            const int n_facets_k = n_facets_per_rank[k];
            const int n_owned_k = global_n_bnd_nodes[k];
            const int n_ghosts_k = n_ghosts_per_rank[k];
            
            // Local coordinates for this rank
            const double* local_coords = coords_ptrs[rank];
            
            for (int j = 0; j < n_facets_k; ++j) {
                // Extract facet node indices (direct array access)
                std::array<int, 3> facet_nodes = {
                    foreign_facets[j*3 + 0],
                    foreign_facets[j*3 + 1],
                    foreign_facets[j*3 + 2]
                };
                for (int l = 0; l < 3; ++l) {
                    if (facet_nodes[l] < 0 || facet_nodes[l] >= n_nodes_k) {
                        throw std::runtime_error(
                            "Facet node index out of bounds on rank " + std::to_string(k) +
                            ": node=" + std::to_string(facet_nodes[l]) +
                            ", n_nodes=" + std::to_string(n_nodes_k)
                        );
                    }
                }
                
                // Map facet nodes to global indices
                std::array<int, 3> facet_nodes_index;
                for (int l = 0; l < 3; ++l) {
                    if (facet_nodes[l] < n_owned_k) {
                        facet_nodes_index[l] = bnd_local_to_bnd_global_direct(k, facet_nodes[l], global_n_bnd_nodes, size);
                    } else {
                        // ghost_idx indexes row in ghost table (matches bnd_nodes[n_owned+ghost_idx]).
                        int ghost_idx = facet_nodes[l] - n_owned_k;
                        if (ghost_idx < 0 || ghost_idx >= n_ghosts_k) {
                            throw std::runtime_error(
                                "Ghost index out of bounds on rank " + std::to_string(k) +
                                ": ghost_idx=" + std::to_string(ghost_idx) +
                                ", n_ghosts=" + std::to_string(n_ghosts_k)
                            );
                        }
                        int owner = foreign_ghosts[ghost_idx*2 + 0];
                        int loc_bnd_index = foreign_ghosts[ghost_idx*2 + 1];
                        if (owner < 0 || owner >= size) {
                            throw std::runtime_error(
                                "Ghost owner out of bounds on rank " + std::to_string(k) +
                                ": owner=" + std::to_string(owner) +
                                ", size=" + std::to_string(size)
                            );
                        }
                        if (loc_bnd_index < 0 || loc_bnd_index >= global_n_bnd_nodes[owner]) {
                            throw std::runtime_error(
                                "Ghost local boundary index out of bounds on owner " + std::to_string(owner) +
                                ": idx=" + std::to_string(loc_bnd_index) +
                                ", n_owned_owner=" + std::to_string(global_n_bnd_nodes[owner])
                            );
                        }
                        facet_nodes_index[l] = bnd_local_to_bnd_global_direct(owner, loc_bnd_index, global_n_bnd_nodes, size);
                    }
                }
                
                // Compute BEM element for each local row
                for (int i = 0; i < n_local; ++i) {
                    int global_i = bnd_local_to_bnd_global_direct(rank, i, global_n_bnd_nodes, size);
                    
                    // Skip if node is part of the facet (diagonal-like term)
                    if (std::find(facet_nodes_index.begin(), facet_nodes_index.end(), global_i) 
                        != facet_nodes_index.end()) {
                        continue;
                    }
                    
                    // Extract coordinates (direct pointer access, cache-friendly)
                    Vector3D b_vert = {
                        local_coords[i*3 + 0],
                        local_coords[i*3 + 1],
                        local_coords[i*3 + 2]
                    };
                    
                    Vector3D facv1 = {
                        foreign_coords[facet_nodes[0]*3 + 0],
                        foreign_coords[facet_nodes[0]*3 + 1],
                        foreign_coords[facet_nodes[0]*3 + 2]
                    };
                    
                    Vector3D facv2 = {
                        foreign_coords[facet_nodes[1]*3 + 0],
                        foreign_coords[facet_nodes[1]*3 + 1],
                        foreign_coords[facet_nodes[1]*3 + 2]
                    };
                    
                    Vector3D facv3 = {
                        foreign_coords[facet_nodes[2]*3 + 0],
                        foreign_coords[facet_nodes[2]*3 + 1],
                        foreign_coords[facet_nodes[2]*3 + 2]
                    };
                    
                    // Compute BEM element contribution
                    Vector3D tmp_bele = compute_belement(b_vert, facv1, facv2, facv3);
                    
                    // Accumulate to output matrix
                    for (int l = 0; l < 3; ++l) {
                        add_entry(i, facet_nodes_index[l], tmp_bele[l]);
                    }
                }
            }
        }
        
        // Diagonal: B_ii = solid_angle_i / (4*pi) - 1 (computed in Python/C++ solid-angle pass).
        if (solid_angles_ptr != nullptr && solid_angles_size > 0) {
            if (static_cast<int>(solid_angles_size) != n_local) {
                std::cerr << "Warning: solid_angles size (" << solid_angles_size 
                          << ") doesn't match n_local (" << n_local << ")" << std::endl;
            }
            
            constexpr double inv_4pi = 1.0 / (4.0 * M_PI);
            for (int i = 0; i < n_local && i < static_cast<int>(solid_angles_size); ++i) {
                int idx_global = bnd_local_to_bnd_global_direct(rank, i, global_n_bnd_nodes, size);
                set_entry(i, idx_global, solid_angles_ptr[i] * inv_4pi - 1.0);
            }
        }
        
        if (use_output_buffer) {
            return Matrix();
        }
        return B;
    }
    
    /**
     * STANDARD API (CONVENIENCE)
     * Assemble the BEM matrix for this rank with optional pre-computed solid angles
     * 
     * @param solid_angles_ptr: Pointer to pre-computed solid angles array (default nullptr)
     * @param solid_angles_size: Size of the solid angles array (default 0)
     * 
     * If solid_angles_ptr is not nullptr, diagonal terms are set as: omega/(4π) - 1
     * Otherwise, diagonal terms are left as computed from off-diagonal assembly
     */
    inline Matrix assemble(
        int rank,
        int size,
        const std::vector<BoundaryData>& bnd_data_per_rank,
        const std::vector<int>& global_n_bnd_nodes,
        const std::vector<int>& bnd_nodes,
        const MeshTopology& mesh_topology,
        const double* solid_angles_ptr = nullptr,
        size_t solid_angles_size = 0
    ) {
        int n_local = global_n_bnd_nodes[rank];
        int n_global = std::accumulate(global_n_bnd_nodes.begin(), global_n_bnd_nodes.end(), 0);
        
        const auto& bnd_nodes_coordinates = bnd_data_per_rank[rank].bnd_nodes_coordinates;
        
        Matrix B(n_local, std::vector<double>(n_global, 0.0));
        
        // Off-diagonal terms
        
        for (int k = 0; k < size; ++k) {
            const auto& data = bnd_data_per_rank[k];
            const auto& foreign_ghost_bnd_nodes_indices = data.ghost_bnd_nodes_indices;
            const auto& foreign_bnd_nodes_coordinates = data.bnd_nodes_coordinates;
            const auto& foreign_bnd_f_to_bnd_n = data.bnd_f_to_bnd_n;
            
            for (size_t j = 0; j < foreign_bnd_f_to_bnd_n.size(); ++j) {
                const auto& facet_nodes = foreign_bnd_f_to_bnd_n[j];
                
                std::array<int, 3> facet_nodes_index;
                for (int l = 0; l < 3; ++l) {
                    if (facet_nodes[l] < global_n_bnd_nodes[k]) {
                        facet_nodes_index[l] = bnd_local_to_bnd_global(k, facet_nodes[l], global_n_bnd_nodes);
                    } else {
                        auto [owner, loc_bnd_index] = 
                            foreign_ghost_bnd_nodes_indices[facet_nodes[l] - global_n_bnd_nodes[k]];
                        facet_nodes_index[l] = bnd_local_to_bnd_global(owner, loc_bnd_index, global_n_bnd_nodes);
                    }
                }
                
                for (int i = 0; i < n_local; ++i) {
                    int global_i = bnd_local_to_bnd_global(rank, i, global_n_bnd_nodes);
                    if (std::find(facet_nodes_index.begin(), facet_nodes_index.end(), global_i) 
                        != facet_nodes_index.end()) {
                        continue;
                    }
                    
                    const Vector3D& b_vert = bnd_nodes_coordinates[i];
                    const Vector3D& facv1 = foreign_bnd_nodes_coordinates[facet_nodes[0]];
                    const Vector3D& facv2 = foreign_bnd_nodes_coordinates[facet_nodes[1]];
                    const Vector3D& facv3 = foreign_bnd_nodes_coordinates[facet_nodes[2]];
                    
                    Vector3D tmp_bele = compute_belement(b_vert, facv1, facv2, facv3);
                    
                    for (int l = 0; l < 3; ++l) {
                        B[i][facet_nodes_index[l]] += tmp_bele[l];
                    }
                }
            }
        }
        
        // Diagonal terms using pre-computed solid angles (via pointer, zero-copy)
        if (solid_angles_ptr != nullptr && solid_angles_size > 0) {
            // Verify we have the right number of solid angles
            if (static_cast<int>(solid_angles_size) != n_local) {
                std::cerr << "Warning: solid_angles size (" << solid_angles_size 
                          << ") doesn't match n_local (" << n_local << ")" << std::endl;
            }
            
            // Set diagonal entries using pre-computed solid angles (direct pointer access - zero-copy!)
            constexpr double inv_4pi = 1.0 / (4.0 * M_PI);
            for (int i = 0; i < n_local && i < static_cast<int>(solid_angles_size); ++i) {
                int idx_global = bnd_local_to_bnd_global(rank, i, global_n_bnd_nodes);
                B[i][idx_global] = solid_angles_ptr[i] * inv_4pi - 1.0;
            }
        }
        
        return B;
    }
    
    /**
     * Save matrix to binary file
     */
    inline void save_matrix(const std::string& filename, const Matrix& mat) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        size_t rows = mat.size();
        size_t cols = rows > 0 ? mat[0].size() : 0;
        
        out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
        
        for (const auto& row : mat) {
            out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));
        }
        
        out.close();
    }
    
    /**
     * Save matrix to ASCII file
     */
    inline void save_matrix_ascii(const std::string& filename, const Matrix& mat) {
        std::ofstream out(filename);
        if (!out) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        size_t rows = mat.size();
        size_t cols = rows > 0 ? mat[0].size() : 0;
        
        out << std::scientific << std::setprecision(16);
        out << rows << " " << cols << "\n";
        
        for (const auto& row : mat) {
            for (size_t j = 0; j < row.size(); ++j) {
                out << row[j];
                if (j < row.size() - 1) out << " ";
            }
            out << "\n";
        }
        
        out.close();
    }
    
    /**
     * Load matrix from binary file
     */
    inline Matrix load_matrix(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return Matrix();
        }
        
        size_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        
        Matrix mat(rows, std::vector<double>(cols));
        for (auto& row : mat) {
            in.read(reinterpret_cast<char*>(row.data()), cols * sizeof(double));
        }
        
        in.close();
        return mat;
    }
    
    /**
     * Load matrix from ASCII file
     */
    inline Matrix load_matrix_ascii(const std::string& filename) {
        std::ifstream in(filename);
        if (!in) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return Matrix();
        }
        
        size_t rows, cols;
        in >> rows >> cols;
        
        Matrix mat(rows, std::vector<double>(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                in >> mat[i][j];
            }
        }
        
        in.close();
        std::cout << "Loaded " << rows << "x" << cols << " matrix from " << filename << std::endl;
        return mat;
    }
    
    /**
     * Compare two matrices
     */
    inline bool compare_matrices(
        const Matrix& A, const Matrix& B, 
        double rtol, double atol,
        double& max_diff, double& avg_diff, size_t& num_mismatches
    ) {
        if (A.size() != B.size() || (A.size() > 0 && A[0].size() != B[0].size())) {
            std::cerr << "Matrix size mismatch: A is " << A.size() << "x" 
                      << (A.size() > 0 ? A[0].size() : 0) << ", B is " << B.size() << "x" 
                      << (B.size() > 0 ? B[0].size() : 0) << std::endl;
            return false;
        }
        
        max_diff = 0.0;
        avg_diff = 0.0;
        num_mismatches = 0;
        size_t total_elements = 0;
        bool passed = true;
        
        for (size_t i = 0; i < A.size(); ++i) {
            for (size_t j = 0; j < A[i].size(); ++j) {
                double diff = std::abs(A[i][j] - B[i][j]);
                double threshold = atol + rtol * std::abs(B[i][j]);
                max_diff = std::max(max_diff, diff);
                avg_diff += diff;
                total_elements++;
                
                if (diff > threshold) {
                    num_mismatches++;
                    passed = false;
                    
                    // Print first few mismatches
                    if (num_mismatches <= 5) {
                        std::cout << "  Mismatch at [" << i << "," << j << "]: "
                                  << "computed=" << A[i][j] << ", reference=" << B[i][j]
                                  << ", diff=" << diff << ", threshold=" << threshold << std::endl;
                    }
                }
            }
        }
        
        if (total_elements > 0) {
            avg_diff /= total_elements;
        }
        
        return passed;
    }

private:
    /**
     * Compute BEM element contribution (exact C implementation from belement.c)
     * This matches the Bele function used in the Python version via CFFI
     */
    inline Vector3D compute_belement(
        const Vector3D& b_vert,
        const Vector3D& facv1,
        const Vector3D& facv2,
        const Vector3D& facv3
    ) {
        constexpr double D_EPS = 1e-14;
        constexpr double PETSC_PI = 3.1415926535897932384626433832795L;
        
        Vector3D matele = {0.0, 0.0, 0.0};
        
        // Copy vertices
        Vector3D rho1 = facv1;
        Vector3D rho2 = facv2;
        Vector3D rho3 = facv3;
        
        // Calculate edge vectors xi_j
        Vector3D xi1 = subtract(rho2, rho1);  // xi1 = rho2 - rho1
        Vector3D xi2 = subtract(rho3, rho2);  // xi2 = rho3 - rho2
        Vector3D xi3 = subtract(rho1, rho3);  // xi3 = rho1 - rho3
        
        // Calculate zeta direction (normal)
        Vector3D zeta = cross(xi1, xi2);
        
        // Calculate area of triangle
        double zetal = norm(zeta);
        double a = 0.5 * zetal;
        
        // Renormalize zeta
        if (zetal > D_EPS) {
            zeta = {zeta[0] / zetal, zeta[1] / zetal, zeta[2] / zetal};
        } else {
            return {0.0, 0.0, 0.0};
        }
        
        // Calculate s_j and normalize xi_j
        double s1 = norm(xi1);
        if (s1 > D_EPS) {
            xi1 = {xi1[0] / s1, xi1[1] / s1, xi1[2] / s1};
        } else {
            return {0.0, 0.0, 0.0};
        }
        
        double s2 = norm(xi2);
        if (s2 > D_EPS) {
            xi2 = {xi2[0] / s2, xi2[1] / s2, xi2[2] / s2};
        } else {
            return {0.0, 0.0, 0.0};
        }
        
        double s3 = norm(xi3);
        if (s3 > D_EPS) {
            xi3 = {xi3[0] / s3, xi3[1] / s3, xi3[2] / s3};
        } else {
            return {0.0, 0.0, 0.0};
        }
        
        // Calculate eta_j = zeta x xi_j
        Vector3D eta1 = cross(zeta, xi1);
        Vector3D eta2 = cross(zeta, xi2);
        Vector3D eta3 = cross(zeta, xi3);
        
        // Calculate gamma matrix elements
        Vector3D gamma1, gamma2, gamma3;
        gamma1[0] = dot(xi2, xi1);  // gamma3[1] also
        gamma1[1] = dot(xi2, xi2);
        gamma1[2] = dot(xi2, xi3);  // gamma2[1] also
        
        gamma2[0] = dot(xi3, xi1);  // gamma3[2] also
        gamma2[1] = gamma1[2];
        gamma2[2] = dot(xi3, xi3);
        
        gamma3[0] = dot(xi1, xi1);
        gamma3[1] = gamma1[0];
        gamma3[2] = gamma2[0];
        
        // Check if point is in plane (distance from plane) - before modifying vertices
        // This implements PointFromPlane from belement.c
        Vector3D ab = subtract(rho1, rho2);
        Vector3D ac = subtract(rho1, rho3);
        Vector3D plane_normal = cross(ab, ac);
        Vector3D x_minus_v1 = subtract(b_vert, rho1);
        double d = dot(x_minus_v1, plane_normal);
        
        if (std::abs(d) < D_EPS) {
            return {0.0, 0.0, 0.0};
        }
        
        // Calculate rho_j = vertex_j - b_vert
        rho1 = subtract(rho1, b_vert);
        rho2 = subtract(rho2, b_vert);
        rho3 = subtract(rho3, b_vert);
        
        // zetal gives normal distance of R from plane
        zetal = dot(zeta, rho1);
        
        // Skip if zetal == 0 (R in plane of triangle)
        if (std::abs(zetal) <= D_EPS) {
            return {0.0, 0.0, 0.0};
        }
        
        double rho1l = norm(rho1);
        double rho2l = norm(rho2);
        double rho3l = norm(rho3);
        
        // Calculate solid angle omega
        double t_nom = rho1l * rho2l * rho3l +
                       rho1l * dot(rho2, rho3) +
                       rho2l * dot(rho3, rho1) +
                       rho3l * dot(rho1, rho2);
        
        double t_denom = std::sqrt(2.0 *
            (rho2l * rho3l + dot(rho2, rho3)) *
            (rho3l * rho1l + dot(rho3, rho1)) *
            (rho1l * rho2l + dot(rho1, rho2))
        );
        
        double omega;
        if (t_denom < D_EPS) {
            return {0.0, 0.0, 0.0};
        }
        
        double ratio = t_nom / t_denom;
        
        // Catch special cases where argument of acos is outside [-1, 1]
        if (ratio < -1.0) {
            omega = (zetal >= 0.0 ? 1.0 : -1.0) * 2.0 * M_PI;
        } else if (ratio > 1.0) {
            return {0.0, 0.0, 0.0};
        } else {
            omega = (zetal >= 0.0 ? 1.0 : -1.0) * 2.0 * std::acos(ratio);
        }
        
        // Calculate eta projections
        double eta1l = dot(eta1, rho1);
        double eta2l = dot(eta2, rho2);
        double eta3l = dot(eta3, rho3);
        
        // Calculate p array (logarithmic terms)
        Vector3D p;
        p[0] = std::log((rho1l + rho2l + s1) / (rho1l + rho2l - s1));
        p[1] = std::log((rho2l + rho3l + s2) / (rho2l + rho3l - s2));
        p[2] = std::log((rho3l + rho1l + s3) / (rho3l + rho1l - s3));
        
        // Safety check for NaN/Inf in p array
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
            return {0.0, 0.0, 0.0};
        }
        
        // Calculate final matrix elements
        matele[0] = (eta2l * omega - zetal * dot(gamma1, p)) * s2 / (8.0 * PETSC_PI * a);
        matele[1] = (eta3l * omega - zetal * dot(gamma2, p)) * s3 / (8.0 * PETSC_PI * a);
        matele[2] = (eta1l * omega - zetal * dot(gamma3, p)) * s1 / (8.0 * PETSC_PI * a);
        
        // Safety check for final result
        if (!std::isfinite(matele[0]) || !std::isfinite(matele[1]) || !std::isfinite(matele[2])) {
            return {0.0, 0.0, 0.0};
        }
        
        return matele;
    }
    
public:
    /**
     * Compute solid angles for all local boundary nodes from local cells
     * Returns partial solid angles (need MPI scatter_reverse to get complete values)
     * 
     * @return Vector of solid angles (one per local boundary node)
     */
    inline std::vector<double> compute_solid_angles_local(
        int rank,
        const std::vector<int>& topology_bnd_nodes,
        const std::vector<std::vector<int>>& node_to_cell,
        const std::vector<std::vector<int>>& cell_to_node,
        const std::vector<Vector3D>& topology_coords
    ) {
        std::vector<double> solid_angles(topology_bnd_nodes.size(), 0.0);
        
        for (size_t i = 0; i < topology_bnd_nodes.size(); ++i) {
            int global_topo_node = topology_bnd_nodes[i];
            
            if (global_topo_node >= static_cast<int>(node_to_cell.size())) {
                continue;
            }
            
            const auto& cells = node_to_cell[global_topo_node];
            
            for (int cell : cells) {
                // Skip cells not in this rank's cell_to_node list
                if (cell < 0 || cell >= static_cast<int>(cell_to_node.size())) {
                    continue;
                }
                
                const auto& cell_nodes = cell_to_node[cell];
                
                // Verify this node is in the cell
                if (std::find(cell_nodes.begin(), cell_nodes.end(), global_topo_node) 
                    == cell_nodes.end()) {
                    continue;
                }
                
                // Get the 3 other nodes (tetrahedron has 4 nodes)
                std::vector<int> other_nodes;
                for (int node : cell_nodes) {
                    if (node != global_topo_node) {
                        other_nodes.push_back(node);
                    }
                }
                
                if (other_nodes.size() != 3) {
                    continue;
                }
                
                // Verify all nodes have valid coordinates
                bool valid = true;
                for (int node : other_nodes) {
                    if (node < 0 || node >= static_cast<int>(topology_coords.size())) {
                        valid = false;
                        break;
                    }
                }
                if (!valid || global_topo_node >= static_cast<int>(topology_coords.size())) {
                    continue;
                }
                
                // Compute solid angle
                const Vector3D& x = topology_coords[global_topo_node];
                const Vector3D& v1 = topology_coords[other_nodes[0]];
                const Vector3D& v2 = topology_coords[other_nodes[1]];
                const Vector3D& v3 = topology_coords[other_nodes[2]];
                
                double omega = compute_solid_angle(x, v1, v2, v3);
                solid_angles[i] += omega;
            }
        }
        
        return solid_angles;
    }
    
    /**
     * Compute solid angle subtended by a triangle at a point
     * Based on the SolidAngle function from belement.c
     */
    inline double compute_solid_angle(
        const Vector3D& x,      // observation point
        const Vector3D& v1,     // triangle vertex 1
        const Vector3D& v2,     // triangle vertex 2
        const Vector3D& v3      // triangle vertex 3
    ) {
        constexpr double D_EPS = 1e-14;
        constexpr double PETSC_PI = 3.1415926535897932384626433832795L;
        
        // Check if point is in plane of triangle
        Vector3D ab = subtract(v1, v2);
        Vector3D ac = subtract(v1, v3);
        Vector3D plane_normal = cross(ab, ac);
        Vector3D x_minus_v1 = subtract(x, v1);
        double d = dot(x_minus_v1, plane_normal);
        
        if (std::abs(d) < D_EPS) {
            return 0.0;
        }
        
        // Calculate edge vectors from x to each vertex
        Vector3D ea = subtract(v1, x);
        Vector3D eb = subtract(v2, x);
        Vector3D ec = subtract(v3, x);
        
        // Calculate normal vectors to planes formed by pairs of edges
        Vector3D nab = cross(ea, eb);
        Vector3D nbc = cross(eb, ec);
        Vector3D nca = cross(ec, ea);
        
        // Normalize the normal vectors
        double nab_norm = norm(nab);
        if (nab_norm < D_EPS) return 0.0;
        nab = {nab[0]/nab_norm, nab[1]/nab_norm, nab[2]/nab_norm};
        
        double nbc_norm = norm(nbc);
        if (nbc_norm < D_EPS) return 0.0;
        nbc = {nbc[0]/nbc_norm, nbc[1]/nbc_norm, nbc[2]/nbc_norm};
        
        double nca_norm = norm(nca);
        if (nca_norm < D_EPS) return 0.0;
        nca = {nca[0]/nca_norm, nca[1]/nca_norm, nca[2]/nca_norm};
        
        // Calculate dihedral angles between facets
        double a_abbc = dot(nab, nbc);
        double a_bcca = dot(nbc, nca);
        double a_caab = dot(nca, nab);
        
        // Handle edge cases for acos argument
        // Matches belement.c lines 267-269 exactly
        auto safe_angle = [](double val) -> double {
            if (val > 1.0) return PETSC_PI;         // angle = π - acos(1) = π (C: t_a_abbc=PETSC_PI)
            if (val < -1.0) return 0.0;             // angle = π - acos(-1) = 0 (C: t_a_abbc=0)
            return PETSC_PI - std::acos(val);       // angle = π - acos(val)
        };
        
        double angle_abbc = safe_angle(a_abbc);
        double angle_bcca = safe_angle(a_bcca);
        double angle_caab = safe_angle(a_caab);
        
        // Solid angle = sum of dihedral angles - π
        double omega = angle_abbc + angle_bcca + angle_caab - PETSC_PI;
        
        return omega;
    }
    
    
    /**
     * Convert local boundary node index to global index
     */
    inline int bnd_local_to_bnd_global(int k, int i, const std::vector<int>& global_n_bnd_nodes) {
        return std::accumulate(global_n_bnd_nodes.begin(), global_n_bnd_nodes.begin() + k, 0) + i;
    }
    
    /**
     * Convert local boundary node index to global index (direct pointer API)
     */
    inline int bnd_local_to_bnd_global_direct(int k, int i, const int* global_n_bnd_nodes, int size) {
        int offset = 0;
        for (int rank = 0; rank < k; ++rank) {
            offset += global_n_bnd_nodes[rank];
        }
        return offset + i;
    }
    
    // Vector operations
    inline Vector3D subtract(const Vector3D& a, const Vector3D& b) {
        return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }
    
    inline Vector3D cross(const Vector3D& a, const Vector3D& b) {
        return {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }
    
    inline double dot(const Vector3D& a, const Vector3D& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    
    inline double norm(const Vector3D& v) {
        return std::sqrt(dot(v, v));
    }
    
    inline Vector3D normalize(const Vector3D& v) {
        double n = norm(v);
        if (n < 1e-15) {
            return {0.0, 0.0, 0.0};
        }
        return {v[0] / n, v[1] / n, v[2] / n};
    }
};

} // namespace bem
