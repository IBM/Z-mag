#pragma once

#include <vector>
#include <string>
#include <array>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace bem {

using Vector3D = std::array<double, 3>;

// Structure to hold boundary data
struct BoundaryData {
    std::vector<std::pair<int, int>> ghost_bnd_nodes_indices;
    std::vector<Vector3D> bnd_nodes_coordinates;
    std::vector<std::array<int, 3>> bnd_f_to_bnd_n;
    int n_bnd_nodes;
};

// Structure to hold mesh topology
struct MeshTopology {
    std::vector<std::vector<int>> node_to_cell;
    std::vector<std::vector<int>> cell_to_node;
    std::vector<Vector3D> topology_nodes_coordinates;
    std::vector<int> topology_bnd_nodes;
};

/**
 * Data loader class for BEM input data (Header-only implementation)
 */
class DataLoader {
public:
    DataLoader() = default;
    
    // Load boundary data from ASCII files for all ranks
    inline bool load_boundary_data_ascii(
        int rank, int size,
        const std::string& data_dir,
        std::vector<BoundaryData>& bnd_data_per_rank,
        std::vector<int>& global_n_bnd_nodes
    ) {
        try {
            bnd_data_per_rank.clear();
            global_n_bnd_nodes.clear();
            
            // Load boundary data for all ranks
            for (int r = 0; r < size; ++r) {
                std::string filename = data_dir + "/bnd_data_rank" + std::to_string(r) + ".txt";
                
                std::cout << "[Rank " << rank << "] Loading: " << filename << std::endl;
                
                std::ifstream in(filename);
                if (!in) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }
                
                BoundaryData local_data;
                
                // Read n_bnd_nodes
                in >> local_data.n_bnd_nodes;
                
                // Read ghost_bnd_nodes_indices
                int n_ghost;
                in >> n_ghost;
                local_data.ghost_bnd_nodes_indices.resize(n_ghost);
                for (int i = 0; i < n_ghost; ++i) {
                    int owner, local_idx;
                    in >> owner >> local_idx;
                    local_data.ghost_bnd_nodes_indices[i] = {owner, local_idx};
                }
                
                // Read bnd_nodes_coordinates
                int n_coords;
                in >> n_coords;
                local_data.bnd_nodes_coordinates.resize(n_coords);
                for (int i = 0; i < n_coords; ++i) {
                    in >> local_data.bnd_nodes_coordinates[i][0]
                       >> local_data.bnd_nodes_coordinates[i][1]
                       >> local_data.bnd_nodes_coordinates[i][2];
                }
                
                // Read bnd_f_to_bnd_n
                int n_facets;
                in >> n_facets;
                local_data.bnd_f_to_bnd_n.resize(n_facets);
                for (int i = 0; i < n_facets; ++i) {
                    in >> local_data.bnd_f_to_bnd_n[i][0]
                       >> local_data.bnd_f_to_bnd_n[i][1]
                       >> local_data.bnd_f_to_bnd_n[i][2];
                }
                
                in.close();
                
                bnd_data_per_rank.push_back(local_data);
                global_n_bnd_nodes.push_back(local_data.n_bnd_nodes);
            }
            
            std::cout << "[Rank " << rank << "] Successfully loaded data for " << size << " ranks" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank << "] ERROR loading boundary data: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Load mesh topology from ASCII file
    inline bool load_mesh_topology_ascii(
        int rank,
        const std::string& data_dir,
        MeshTopology& mesh_topology
    ) {
        try {
            std::string filename = data_dir + "/mesh_topology_rank" + std::to_string(rank) + ".txt";
            std::cout << "[Rank " << rank << "] Loading mesh topology: " << filename << std::endl;
            
            std::ifstream in(filename);
            if (!in) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            
            // Read node_to_cell
            int n_nodes;
            in >> n_nodes;
            mesh_topology.node_to_cell.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                int n_cells;
                in >> n_cells;
                mesh_topology.node_to_cell[i].resize(n_cells);
                for (int j = 0; j < n_cells; ++j) {
                    in >> mesh_topology.node_to_cell[i][j];
                }
            }
            
            // Read cell_to_node
            int n_cells;
            in >> n_cells;
            mesh_topology.cell_to_node.resize(n_cells);
            for (int i = 0; i < n_cells; ++i) {
                int n_nodes_in_cell;
                in >> n_nodes_in_cell;
                mesh_topology.cell_to_node[i].resize(n_nodes_in_cell);
                for (int j = 0; j < n_nodes_in_cell; ++j) {
                    in >> mesh_topology.cell_to_node[i][j];
                }
            }
            
            // Read topology_nodes_coordinates
            int n_coords;
            in >> n_coords;
            mesh_topology.topology_nodes_coordinates.resize(n_coords);
            for (int i = 0; i < n_coords; ++i) {
                in >> mesh_topology.topology_nodes_coordinates[i][0]
                   >> mesh_topology.topology_nodes_coordinates[i][1]
                   >> mesh_topology.topology_nodes_coordinates[i][2];
            }
            
            // Read topology_bnd_nodes
            int n_bnd;
            in >> n_bnd;
            mesh_topology.topology_bnd_nodes.resize(n_bnd);
            for (int i = 0; i < n_bnd; ++i) {
                in >> mesh_topology.topology_bnd_nodes[i];
            }
            
            in.close();
            
            std::cout << "[Rank " << rank << "] Successfully loaded mesh topology" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank << "] ERROR loading mesh topology: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Load boundary nodes from ASCII file
    inline bool load_boundary_nodes_ascii(
        int rank,
        const std::string& data_dir,
        std::vector<int>& bnd_nodes
    ) {
        try {
            std::string filename = data_dir + "/bnd_nodes_rank" + std::to_string(rank) + ".txt";
            std::cout << "[Rank " << rank << "] Loading boundary nodes: " << filename << std::endl;
            
            std::ifstream in(filename);
            if (!in) {
                throw std::runtime_error("Cannot open file: " + filename);
            }
            
            int n_nodes;
            in >> n_nodes;
            bnd_nodes.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                in >> bnd_nodes[i];
            }
            
            in.close();
            
            std::cout << "[Rank " << rank << "] Successfully loaded " << bnd_nodes.size() << " boundary nodes" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[Rank " << rank << "] ERROR loading boundary nodes: " << e.what() << std::endl;
            return false;
        }
    }
    
    
    // Dump loaded data to ASCII for verification
    inline void dump_bnd_data_ascii(
        const std::string& filename,
        const BoundaryData& data,
        int n_bnd_nodes
    ) {
        std::ofstream out(filename);
        if (!out) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        out << std::scientific << std::setprecision(16);
        
        // n_bnd_nodes
        out << n_bnd_nodes << "\n";
        
        // ghost_bnd_nodes_indices
        out << data.ghost_bnd_nodes_indices.size() << "\n";
        for (const auto& [owner, local_idx] : data.ghost_bnd_nodes_indices) {
            out << owner << " " << local_idx << "\n";
        }
        
        // bnd_nodes_coordinates
        out << data.bnd_nodes_coordinates.size() << "\n";
        for (const auto& coord : data.bnd_nodes_coordinates) {
            out << coord[0] << " " << coord[1] << " " << coord[2] << "\n";
        }
        
        // bnd_f_to_bnd_n
        out << data.bnd_f_to_bnd_n.size() << "\n";
        for (const auto& facet : data.bnd_f_to_bnd_n) {
            out << facet[0] << " " << facet[1] << " " << facet[2] << "\n";
        }
        
        out.close();
    }
    
    inline void dump_bnd_nodes_ascii(
        const std::string& filename,
        const std::vector<int>& bnd_nodes
    ) {
        std::ofstream out(filename);
        if (!out) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        out << bnd_nodes.size() << "\n";
        for (int val : bnd_nodes) {
            out << val << "\n";
        }
        
        out.close();
    }
    
    inline void dump_mesh_topology_ascii(
        const std::string& filename,
        const MeshTopology& topo
    ) {
        std::ofstream out(filename);
        if (!out) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }
        
        out << std::scientific << std::setprecision(16);
        
        // node_to_cell
        out << topo.node_to_cell.size() << "\n";
        for (const auto& cells : topo.node_to_cell) {
            out << cells.size();
            for (int cell : cells) {
                out << " " << cell;
            }
            out << "\n";
        }
        
        // cell_to_node
        out << topo.cell_to_node.size() << "\n";
        for (const auto& nodes : topo.cell_to_node) {
            out << nodes.size();
            for (int node : nodes) {
                out << " " << node;
            }
            out << "\n";
        }
        
        // topology_nodes_coordinates
        out << topo.topology_nodes_coordinates.size() << "\n";
        for (const auto& coord : topo.topology_nodes_coordinates) {
            out << coord[0] << " " << coord[1] << " " << coord[2] << "\n";
        }
        
        // topology_bnd_nodes
        out << topo.topology_bnd_nodes.size() << "\n";
        for (int val : topo.topology_bnd_nodes) {
            out << val << "\n";
        }
        
        out.close();
    }
};

} // namespace bem

