"""
Distributed demagnetizing BEM matrix assembly (Fredkin--Koehler formulation).

Each MPI rank stores a dense block of shape ``(n_owned_boundary, n_global_boundary)``.
Off-diagonal entries are boundary facet integrals; diagonal entries use solid angles
from incident tetrahedra (``B_ii = \\Omega_i / (4\\pi) - 1``).

The heavy work is done in pybind11 modules ``bem_cpp`` (production) or
``bem_cpp_debug`` (dict-based API with extra checks). Select via
:func:`configure_cpp_backend` or the environment variable ``BEM_CPP_BACKEND``.
"""

import os
import sys
from functools import partial
from typing import Any, Literal, Optional

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, cpp

CppBackendKind = Literal["production", "debug", "bem_cpp", "bem_cpp_debug"]

_CPP_MODULE_NAMES: dict[str, str] = {
    "production": "bem_cpp",
    "bem_cpp": "bem_cpp",
    "debug": "bem_cpp_debug",
    "bem_cpp_debug": "bem_cpp_debug",
}

_module_dir = os.path.dirname(os.path.abspath(__file__))
_cpp_build_path = os.path.join(_module_dir, "cpp", "build")
if _cpp_build_path not in sys.path:
    sys.path.insert(0, _cpp_build_path)

_cpp_backend_kind: str = "production"
cpp_backend: Any = None
_CPP_AVAILABLE = False
_import_error: str = ""


def _normalize_cpp_backend_kind(kind: str) -> str:
    """Map user-facing backend name to a supported key.

    :param kind: ``production``, ``debug``, ``bem_cpp``, or ``bem_cpp_debug``.
    :returns: Normalized key present in :data:`_CPP_MODULE_NAMES`.
    :raises ValueError: if ``kind`` is not recognized.
    """
    key = kind.strip().lower()
    if key not in _CPP_MODULE_NAMES:
        allowed = ", ".join(sorted(_CPP_MODULE_NAMES))
        raise ValueError(f"Unknown C++ BEM backend '{kind}'. Choose one of: {allowed}.")
    return key


def load_cpp_backend(kind: str = "production"):
    """Import a compiled pybind11 BEM extension module.

    :param kind: ``production`` (``bem_cpp``) or ``debug`` (``bem_cpp_debug``).
    :returns: Tuple ``(module, available, import_error_message)``.
    """
    key = _normalize_cpp_backend_kind(kind)
    module_name = _CPP_MODULE_NAMES[key]
    try:
        import importlib

        module = importlib.import_module(module_name)
        return module, True, ""
    except ImportError as exc:
        return None, False, str(exc)


def configure_cpp_backend(kind: Optional[str] = None) -> str:
    """Select which compiled extension is used for subsequent assembly.

    If ``kind`` is omitted, reads the environment variable ``BEM_CPP_BACKEND``
    (default ``production``).

    :param kind: Optional backend name; see :func:`_normalize_cpp_backend_kind`.
    :returns: Normalized backend key (``production`` or ``debug``).
    """
    global cpp_backend, _CPP_AVAILABLE, _import_error, _cpp_backend_kind

    if kind is None:
        kind = os.environ.get("BEM_CPP_BACKEND", "production")
    key = _normalize_cpp_backend_kind(kind)
    module, available, err = load_cpp_backend(key)
    _cpp_backend_kind = key
    cpp_backend = module
    _CPP_AVAILABLE = available
    _import_error = err
    return key


configure_cpp_backend()


def format_bytes(nbytes: int) -> str:
    """Format a byte count with binary SI units (B, KiB, …, TiB).

    :param nbytes: Non-negative size in bytes.
    :returns: Human-readable string, e.g. ``163 KiB`` or ``18.4 GiB``.
    """
    nbytes = max(0, int(nbytes))
    if nbytes < 1024:
        return f"{nbytes} B"
    for unit, scale in (
        ("TiB", 1 << 40),
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
    ):
        if nbytes >= scale:
            val = nbytes / scale
            if val >= 100:
                return f"{val:.0f} {unit}"
            if val >= 10:
                return f"{val:.1f} {unit}"
            return f"{val:.2f} {unit}"
    return f"{nbytes} B"


def format_bytes_range(nbytes_min: int, nbytes_max: int) -> str:
    """Format a min–max byte range, collapsing to one value when equal."""
    if nbytes_min == nbytes_max:
        return format_bytes(nbytes_min)
    return f"{format_bytes(nbytes_min)}–{format_bytes(nbytes_max)}"


# ---------------------------------------------------------------------------
# Solid-angle helpers (diagonal of the BEM matrix)
# ---------------------------------------------------------------------------

def is_cpp_available():
    """Check if C++ backend is available."""
    return _CPP_AVAILABLE


def get_cpp_backend():
    """Return the active pybind11 module name, or a placeholder if unavailable."""
    if _CPP_AVAILABLE and cpp_backend is not None:
        return cpp_backend.__name__
    return "None - C++ backend not available"


def get_cpp_backend_kind() -> str:
    """Return the normalized backend key (``production`` or ``debug``)."""
    return _cpp_backend_kind


def _compute_solid_angles_cpp(
    rank,
    topology_bnd_nodes,
    mesh_topology,
    domain,
    bnd_nodes,
    n_local_bnd_nodes,
    use_mpi_scatter,
    solid_angle_function,
    backend=None,
):
    """
    Sum solid-angle contributions on boundary nodes (C++ local loop + optional MPI).

    Each rank first accumulates angles from **owned** volume cells only. When
    ``use_mpi_scatter`` is true, partial values are written into a P1 field on
    mesh vertices (geometry node indices) and ``scatter_reverse`` adds
    contributions from neighbouring ranks so shared boundary nodes get the full
    solid angle.
    """
    be = backend if backend is not None else cpp_backend
    if be is None:
        raise RuntimeError("C++ backend requested but not available")

    # Partial solid angles: one value per entry in bnd_nodes (owned + ghost slots).
    partial_solid_angles = be.compute_solid_angles_local(
        rank, topology_bnd_nodes, mesh_topology
    )

    if not use_mpi_scatter:
        return partial_solid_angles[:n_local_bnd_nodes]

    # P1 field on all mesh vertices; scatter_reverse sums over shared ghosts.
    if solid_angle_function is None:
        V_nodes = fem.functionspace(domain, ("CG", 1))
        solid_angle_function = fem.Function(V_nodes)
    else:
        solid_angle_function.x.array[:] = 0.0

    # Scatter partial values into the P1 field at geometry node indices from bnd_nodes.
    for i in range(len(bnd_nodes)):
        solid_angle_function.x.array[bnd_nodes[i]] = partial_solid_angles[i]

    solid_angle_function.x.scatter_reverse(cpp.la.InsertMode.add)

    solid_angles = np.zeros(n_local_bnd_nodes, dtype=np.float64)
    for i in range(n_local_bnd_nodes):
        solid_angles[i] = solid_angle_function.x.array[bnd_nodes[i]]

    return solid_angles


def compute_solid_angles(
    domain,
    bnd_nodes,
    n_local_bnd_nodes,
    topology_bnd_nodes,
    mesh_topology,
    use_mpi_scatter=True,
    solid_angle_function=None,
    backend=None,
):
    """
    Compute solid angles for boundary nodes from incident tetrahedral cells using C++.
    
    This function uses the C++ backend to compute solid angles from local cells,
    then uses MPI scatter_reverse to sum contributions from all ranks.
    
    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        The mesh domain (needed for MPI scatter).
    bnd_nodes : np.ndarray (int)
        Array of boundary node indices (geometry ordering, includes ghosts).
    n_local_bnd_nodes : int
        Number of owned (non-ghost) boundary nodes on this rank.
    topology_bnd_nodes : np.ndarray or list (int)
        Boundary node indices in topology ordering (includes ghosts).
    mesh_topology : dict
        Pre-computed mesh topology with keys:
        - 'node_to_cell': list of lists (node -> incident cells)
        - 'cell_to_node': list of lists (cell -> constituent nodes)
        - 'topology_nodes_coordinates': np.ndarray (coordinates)
        - 'topology_bnd_nodes': np.ndarray (boundary node indices in topology ordering)
    use_mpi_scatter : bool, default True
        If True, use MPI scatter_reverse to sum solid angle contributions
        from ghost cells. Set to False for serial computation.
    solid_angle_function : fem.Function, optional
        Pre-allocated FEniCS function for MPI scatter. If None, will be created.
    backend : module, optional
        pybind11 module (``bem_cpp`` or ``bem_cpp_debug``). Defaults to the active
        module from :func:`configure_cpp_backend`.
    
    Returns
    -------
    np.ndarray (float64)
        Solid angles for the n_local_bnd_nodes owned boundary nodes.
        Shape: (n_local_bnd_nodes,)
    
    Notes
    -----
    - Each rank computes PARTIAL solid angles from LOCAL cells only (C++)
    - MPI scatter_reverse sums contributions from all ranks to get complete values
    - The returned array contains only owned nodes (not ghosts)
    
    Raises
    ------
    RuntimeError
        If C++ backend is not available.
    """
    be = backend if backend is not None else cpp_backend
    if be is None:
        error_msg = (
            "C++ backend is required but not available. "
            "Please compile the C++ backend using: cd cpp && ./install_python_bindings.sh\n"
        )
        if _import_error:
            error_msg += f"Import error details: {_import_error}"
        raise RuntimeError(error_msg)

    comm = domain.comm
    rank = comm.Get_rank()

    return _compute_solid_angles_cpp(
        rank,
        topology_bnd_nodes,
        mesh_topology,
        domain,
        bnd_nodes,
        n_local_bnd_nodes,
        use_mpi_scatter,
        solid_angle_function,
        backend=be,
    )


class BEMAssembler:
    """
    Assemble the MPI-distributed demagnetizing BEM matrix on a DOLFINx mesh.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Magnetized region (tetrahedral cells).
    facet_tags : dolfinx.cpp.mesh.MeshTags_int32
        Facet tags; ``surface_marker`` selects the exterior magnet boundary.
    surface_marker : int
        Tag value for boundary facets carrying BEM unknowns.
    verbose : bool, optional
        If ``True``, rank 0 prints assembly steps and a memory estimate.
    cpp_backend : str, optional
        ``"production"`` (``bem_cpp``) or ``"debug"`` (``bem_cpp_debug``).

    Attributes
    ----------
    BEM : numpy.ndarray
        Local block, shape ``(n_owned_boundary, n_global_boundary)``.
    global_n_bnd_nodes : list of int
        Owned boundary node count on each MPI rank.
    bnd_nodes : numpy.ndarray
        Boundary vertex indices in geometry ordering (includes ghosts).
    gnodes_to_bnodes : numpy.ndarray
        Map geometry vertex index → local boundary index (-1 if not on boundary).
    """

    @staticmethod
    def _gather_payload_nbytes(bnd_data_per_rank: list) -> int:
        """Sum ``nbytes`` of NumPy arrays in the replicated ``allgather`` boundary stack.

        Each MPI rank holds an identical copy of ``bnd_data_per_rank`` after
        :meth:`gather_boundary_data`.

        :param bnd_data_per_rank: List of per-rank boundary dicts from ``allgather``.
        :returns: Total payload size in bytes (host NumPy allocation only).
        """
        total = 0
        for blk in bnd_data_per_rank:
            for key in ("ghost_bnd_nodes_indices", "bnd_nodes_coordinates", "bnd_f_to_bnd_n"):
                arr = blk[key]
                if isinstance(arr, np.ndarray):
                    total += int(arr.nbytes)
        return total

    @staticmethod
    def _mesh_topology_array_nbytes(mesh_topology: dict) -> int:
        """Return ``nbytes`` of the topology coordinate array used for solid angles.

        :param mesh_topology: Mapping containing ``topology_nodes_coordinates``.
        :returns: Byte size of that array, or ``0`` if missing.
        """
        arr = mesh_topology.get("topology_nodes_coordinates")
        if isinstance(arr, np.ndarray):
            return int(arr.nbytes)
        return 0

    def __init__(
        self,
        domain,
        facet_tags,
        surface_marker,
        verbose=True,
        cpp_backend: Optional[str] = None,
    ):
        self.domain = domain
        self.facet_tags = facet_tags
        self.bnd_mrk = surface_marker
        self.comm = domain.comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.verbose = verbose and (self.rank == 0)

        key = (
            _normalize_cpp_backend_kind(cpp_backend)
            if cpp_backend is not None
            else _cpp_backend_kind
        )
        module, available, err = load_cpp_backend(key)
        self._cpp_backend_kind = key
        self._cpp_backend = module
        if not available or module is None:
            raise RuntimeError(
                f"C++ BEM backend '{key}' is not available. "
                f"Build under {_cpp_build_path}. Details: {err}"
            )
        
        self.BEM = None
        self.global_n_bnd_nodes = None
        self.gnodes_to_bnodes = None
        self.bnd_nodes = None
        self.bnd_f_to_bnd_n = None
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("  BEM Assembler Initialized")
            print(f"{'='*70}")
            print(f"MPI ranks: {self.size}")
            print(f"C++ backend: {self._cpp_backend.__name__} ({self._cpp_backend_kind})")
            print(f"Boundary marker: {surface_marker}")
            print(f"{'='*70}\n")
    
    def compute_boundary_nodes(self):
        """
        Identify boundary facets and assign local/global boundary node indices.

        Sets ``bnd_nodes``, ``gnodes_to_bnodes``, ``global_n_bnd_nodes``, and
        ``bnd_f_to_bnd_n`` (facet corners in local boundary numbering).
        """
        if self.verbose:
            print("[Step 1/4] Computing boundary node mapping...")

        n_nodes = self.domain.geometry.index_map().size_local

        # Facets tagged with the magnet/vacuum exterior surface.
        self.bnd_facets = self.facet_tags.find(self.bnd_mrk)

        # Only facets owned by this rank (exclude ghost facets).
        local_bnd_facets = self.bnd_facets[
            self.bnd_facets < self.domain.topology.index_map(self.domain.topology.dim-1).size_local
        ].tolist()
        local_bnd_facets_array = np.array(local_bnd_facets, dtype=np.int32)
        local_bnd_facets_array.flags.writeable = False
        
        # Create topology connectivity (required for entities_to_geometry)
        # Must create connectivity before calling entities_to_geometry
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim-1, 0)  # facet to vertex
        self.domain.topology.create_connectivity(tdim-1, tdim)  # facet to cell
        
        # Create entity permutations (required for oriented operations)
        self.domain.topology.create_entity_permutations()
        
        # Map boundary facets to their constituent nodes
        facet_to_vertex = self.domain.topology.connectivity(self.domain.topology.dim-1, 0)
        
        if local_bnd_facets_array.size == 0:
            # Rank may own no boundary facets; build an empty facet array safely.
            if facet_to_vertex is not None and facet_to_vertex.num_nodes > 0:
                num_vertices_per_facet = len(facet_to_vertex.links(0))
            else:
                if self.verbose:
                    print(f"[Rank {self.rank}] Warning: No facets on this rank")
                num_vertices_per_facet = 0
            bnd_f_to_n = np.empty((0, num_vertices_per_facet), dtype=np.int32)
        else:
            bnd_f_to_n = cpp.mesh.entities_to_geometry(
                self.domain._cpp_object, 
                self.domain.topology.dim-1, 
                local_bnd_facets_array, 
                True
            )
        
        bnd_nodes = list(set(bnd_f_to_n.reshape(-1)))

        # P1 indicator: mark boundary vertices, then communicate ghost layers.
        V_nodes = fem.functionspace(self.domain, ("CG", 1))
        bnd_index_function = fem.Function(V_nodes)
        
        bnd_index_function.x.array[:] = np.zeros(
            self.domain.geometry.index_map().size_local + 
            self.domain.geometry.index_map().num_ghosts
        )
        bnd_index_function.x.array[bnd_nodes] = np.ones(len(bnd_nodes))
        
        bnd_index_function.x.scatter_reverse(cpp.la.InsertMode.add)
        bnd_index_function.x.scatter_forward()

        for i in range(
            self.domain.geometry.index_map().size_local + 
            self.domain.geometry.index_map().num_ghosts
        ):
            if bnd_index_function.x.array[i] != 0 and i not in bnd_nodes:
                bnd_nodes.append(i)
        bnd_nodes.sort()
        
        n_bnd_nodes = len([i for i in bnd_nodes if i < n_nodes])

        # Owned boundary nodes get indices 0 .. n_bnd_nodes-1; others stay -1.
        bnd_index_function.x.array[:] = -np.ones(
            self.domain.geometry.index_map().size_local + 
            self.domain.geometry.index_map().num_ghosts
        )
        bnd_index_function.x.array[bnd_nodes[:n_bnd_nodes]] = range(n_bnd_nodes)
        
        bnd_index_function.x.scatter_forward()

        for i in range(
            self.domain.geometry.index_map().size_local,
            self.domain.geometry.index_map().size_local + 
            self.domain.geometry.index_map().num_ghosts
        ):
            if bnd_index_function.x.array[i] != -1:
                if i not in bnd_nodes:
                    bnd_nodes.append(i)
        
        # Per geometry-ghost slot: local boundary index on owner (-1 if not on bnd).
        self.ghost_bnd_indices = bnd_index_function.x.array[
            self.domain.geometry.index_map().size_local:
        ]

        bnd_nodes.sort()
        self.bnd_nodes = np.array(bnd_nodes, int)
        self.gnodes_to_bnodes = np.array(bnd_index_function.x.array[:], dtype=int)

        for i in range(n_bnd_nodes, self.bnd_nodes.shape[0]):
            self.gnodes_to_bnodes[self.bnd_nodes[i]] = i

        # Facet corners as indices into the local boundary node list.
        self.bnd_f_to_bnd_n = np.zeros(bnd_f_to_n.shape, int)
        for j in range(bnd_f_to_n.shape[0]):
            for l in range(3):
                self.bnd_f_to_bnd_n[j][l] = self.gnodes_to_bnodes[bnd_f_to_n[j][l]]
        
        # global_n_bnd_nodes[k] = #owned boundary nodes on rank k (column offsets).
        nodes_range = self.domain.geometry.index_map().local_range
        local_info = {'nodes_range': nodes_range, 'n_bnd_nodes': n_bnd_nodes}
        all_info = self.comm.allgather(local_info)
        
        self.global_nodes_range = [info['nodes_range'] for info in all_info]
        self.global_n_bnd_nodes = [info['n_bnd_nodes'] for info in all_info]

        if self.verbose:
            n_global = sum(self.global_n_bnd_nodes)
            n_owned_r0 = self.global_n_bnd_nodes[0]
            n_min = min(self.global_n_bnd_nodes)
            n_max = max(self.global_n_bnd_nodes)
            print(f"  ✓ Global boundary nodes (all ranks): {n_global}")
            print(
                f"  ✓ Owned rows per rank: {n_min}–{n_max}  "
                f"(rank 0 owns {n_owned_r0})"
            )

    def gather_boundary_data(self):
        """
        Replicate per-rank boundary facet data on all MPI ranks (``MPI.allgather``).

        Returns
        -------
        list of dict
            One dict per rank with ``ghost_bnd_nodes_indices``, ``bnd_nodes_coordinates``,
            ``bnd_f_to_bnd_n``, and ``n_bnd_nodes``.

        Notes
        -----
        Ghost metadata rows are ordered by boundary ghost index
        ``bnd_nodes[n_owned + k]``, not by raw DOLFINx geometry ghost slots.
        """
        if self.verbose:
            print("[Step 2/4] Gathering boundary data from all ranks...")

        # Facet corners use local boundary indices: owned 0..n_owned-1; ghosts n_owned+k.
        # Row k of ghost_bnd_nodes_indices must match bnd_nodes[n_owned + k] (not geom ghost slot).
        n_owned = int(self.global_n_bnd_nodes[self.rank])
        n_ghost_bnd = int(self.bnd_nodes.shape[0]) - n_owned
        geom_local = self.domain.geometry.index_map().size_local
        n_geom_ghosts = self.domain.geometry.index_map().num_ghosts

        ghost_bnd_nodes_indices = np.zeros((max(n_ghost_bnd, 0), 2), dtype=np.int32)
        for pos in range(n_owned, len(self.bnd_nodes)):
            geom = int(self.bnd_nodes[pos])
            ghost_idx = pos - n_owned
            if geom < geom_local:
                raise RuntimeError(
                    f"BEM ghost packing: expected ghost geometry index >= {geom_local}, got {geom}"
                )
            slot = geom - geom_local
            if slot < 0 or slot >= n_geom_ghosts:
                raise RuntimeError(
                    f"BEM ghost packing: geometry {geom} -> slot {slot} out of ghost range [0,{n_geom_ghosts})"
                )
            owner = int(self.domain.geometry.index_map().owners[slot])
            loc_index = int(self.ghost_bnd_indices[slot])
            ghost_bnd_nodes_indices[ghost_idx, 0] = owner
            ghost_bnd_nodes_indices[ghost_idx, 1] = loc_index
        
        if self.bnd_nodes.shape[0] == 0:
            bnd_nodes_coordinates = np.zeros((0, 3), dtype=np.float64)
        else:
            bnd_nodes_coordinates = np.ascontiguousarray(
                self.domain.geometry.x[self.bnd_nodes], dtype=np.float64
            )
        
        bnd_f_to_bnd_n = np.ascontiguousarray(self.bnd_f_to_bnd_n, dtype=np.int32)

        local_bnd_data = {
            'ghost_bnd_nodes_indices': ghost_bnd_nodes_indices,
            'bnd_nodes_coordinates': bnd_nodes_coordinates,
            'bnd_f_to_bnd_n': bnd_f_to_bnd_n,
            'n_bnd_nodes': self.global_n_bnd_nodes[self.rank]
        }
        
        bnd_data_per_rank = self.comm.allgather(local_bnd_data)

        if self.verbose:
            print(f"  ✓ Gathered boundary data from {len(bnd_data_per_rank)} ranks")

        return bnd_data_per_rank
    
    def bnd_local_to_bnd_global(self, rank_k, local_idx):
        """
        Convert local boundary node index to global BEM matrix index.
        
        Parameters
        ----------
        rank_k : int
            MPI rank owning the node
        local_idx : int
            Local boundary node index on rank_k
        
        Returns
        -------
        int
            Global boundary node index
        """
        return sum(self.global_n_bnd_nodes[:rank_k]) + local_idx

    def _print_memory_estimate_rank0(self, n_bnd_nodes_global: int) -> None:
        """Print a short dense-block RAM estimate on MPI rank 0 (``verbose`` only).

        Each rank stores one block of shape (n_owned, n_global). Labels distinguish
        global DOF counts from per-rank row ownership.

        :param n_bnd_nodes_global: Total boundary nodes / global matrix columns.
        """
        if not self.verbose or self.rank != 0:
            return

        n_owned_r0 = int(self.global_n_bnd_nodes[0])
        bytes_per_rank = [
            int(n_owned) * int(n_bnd_nodes_global) * 8
            for n_owned in self.global_n_bnd_nodes
        ]
        dense_min = min(bytes_per_rank)
        dense_max = max(bytes_per_rank)
        bytes_r0 = bytes_per_rank[0]
        total_dense = sum(bytes_per_rank)

        print("  RAM estimate for Step 4 (one dense block per rank, rows × cols):")
        print(
            f"    Global matrix columns: {n_bnd_nodes_global}  "
            f"(same on every rank)"
        )
        print(
            f"    Per rank: {format_bytes_range(dense_min, dense_max)}  "
            f"|  rank 0: {n_owned_r0}×{n_bnd_nodes_global} → {format_bytes(bytes_r0)}"
        )
        print(
            f"    All ranks (dense blocks only): {format_bytes(total_dense)}  "
            f"(excludes mesh / FEniCS / PETSc)"
        )

    def assemble(self):
        """
        Assemble the local BEM block on this MPI rank.

        Returns
        -------
        numpy.ndarray
            Dense block ``(n_owned_boundary, n_global_boundary)`` in ``self.BEM``.
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("  BEM Matrix Assembly")
            print("=" * 70)

        self.compute_boundary_nodes()
        self.comm.Barrier()

        bnd_data_per_rank = self.gather_boundary_data()
        self.comm.Barrier()

        n_bnd_nodes_global = sum(self.global_n_bnd_nodes)
        n_local = self.global_n_bnd_nodes[self.rank]

        topology_bnd_nodes = self._build_topology_bnd_nodes()
        mesh_topology = self._build_mesh_topology(topology_bnd_nodes)

        self._print_memory_estimate_rank0(n_bnd_nodes_global)

        if self.verbose:
            print("[Step 3/4] Computing solid angles (diagonal terms)...", flush=True)

        solid_angles = compute_solid_angles(
            domain=self.domain,
            bnd_nodes=self.bnd_nodes,
            n_local_bnd_nodes=n_local,
            topology_bnd_nodes=topology_bnd_nodes,
            mesh_topology=mesh_topology,
            use_mpi_scatter=True,
            backend=self._cpp_backend,
        )
        self.comm.Barrier()

        if self.verbose:
            print("[Step 4/4] Assembling BEM matrix (C++ backend)...", flush=True)
            print(
                f"  Local block on rank 0: ({n_local}, {n_bnd_nodes_global})  "
                f"= {format_bytes(int(n_local) * int(n_bnd_nodes_global) * 8)}",
                flush=True,
            )

        if self._cpp_backend is None:
            raise RuntimeError(
                "C++ backend is required but not available. "
                "Please compile the C++ backend using: cd cpp && ./install_python_bindings.sh"
            )

        self.BEM = self._assemble_cpp(bnd_data_per_rank, mesh_topology, solid_angles, n_local, n_bnd_nodes_global)
        
        if self.verbose:
            print(f"  ✓ Assembly complete!")
            print("=" * 70 + "\n")

        return self.BEM
    
    def _assemble_cpp(self, bnd_data_per_rank, mesh_topology, solid_angles, n_local, n_global):
        """
        Fill the local dense block via pybind11.

        ``bem_cpp_debug`` accepts Python dicts per rank; ``bem_cpp`` uses NumPy
        buffers and an optional preallocated ``output_buffer``.
        """
        be = self._cpp_backend
        if self.verbose:
            print(f"  Using C++ backend: {be.__name__}")

        use_dict_api = "debug" in be.__name__.lower()

        if use_dict_api:
            B_local = be.assemble_bem_matrix(
                self.rank,
                self.size,
                bnd_data_per_rank,
                self.bnd_nodes,
                mesh_topology,
                solid_angles,
            )
        else:
            # bem_cpp: zero-copy pointers into contiguous NumPy arrays.
            coords_per_rank = [d['bnd_nodes_coordinates'] for d in bnd_data_per_rank]
            facets_per_rank = [d['bnd_f_to_bnd_n'] for d in bnd_data_per_rank]
            ghost_indices_per_rank = [d['ghost_bnd_nodes_indices'] for d in bnd_data_per_rank]
            global_n_bnd_nodes = np.array([d['n_bnd_nodes'] for d in bnd_data_per_rank], dtype=np.int32)

            # Validate ghost tables and facet indices before the direct C++ kernel.
            for r in range(self.size):
                n_owned_r = int(global_n_bnd_nodes[r])
                n_total_r = int(coords_per_rank[r].shape[0])
                n_ghost_r = max(n_total_r - n_owned_r, 0)

                if ghost_indices_per_rank[r].shape[0] != n_ghost_r:
                    raise RuntimeError(
                        f"Rank {r}: ghost table rows={ghost_indices_per_rank[r].shape[0]} "
                        f"but expected n_ghost={n_ghost_r} from coords({n_total_r})-owned({n_owned_r})."
                    )

                if facets_per_rank[r].size:
                    fmin = int(facets_per_rank[r].min())
                    fmax = int(facets_per_rank[r].max())
                    if fmin < 0 or fmax >= n_total_r:
                        raise RuntimeError(
                            f"Rank {r}: facet boundary index out of bounds [{fmin},{fmax}] for n_total={n_total_r}."
                        )

                for g in range(n_ghost_r):
                    owner = int(ghost_indices_per_rank[r][g, 0])
                    loc_idx = int(ghost_indices_per_rank[r][g, 1])
                    if owner < 0 or owner >= self.size:
                        raise RuntimeError(f"Rank {r}: ghost row {g} has invalid owner={owner}.")
                    if loc_idx < 0 or loc_idx >= int(global_n_bnd_nodes[owner]):
                        raise RuntimeError(
                            f"Rank {r}: ghost row {g} has invalid owner-local index {loc_idx} "
                            f"(owner={owner}, n_owned_owner={int(global_n_bnd_nodes[owner])})."
                        )

            output_buffer = np.zeros((n_local, n_global), dtype=np.float64)

            # Direct API requires C-contiguous float64 / int32 buffers.
            for i in range(len(coords_per_rank)):
                if not be.is_c_contiguous(coords_per_rank[i]):
                    coords_per_rank[i] = np.ascontiguousarray(coords_per_rank[i])
                if not be.is_c_contiguous(facets_per_rank[i]):
                    facets_per_rank[i] = np.ascontiguousarray(facets_per_rank[i])
                if not be.is_c_contiguous(ghost_indices_per_rank[i]):
                    ghost_indices_per_rank[i] = np.ascontiguousarray(ghost_indices_per_rank[i])

            B_local = be.assemble_bem_matrix(
                self.rank,
                self.size,
                coords_per_rank,
                facets_per_rank,
                ghost_indices_per_rank,
                global_n_bnd_nodes,
                output_buffer=output_buffer,
                solid_angles=solid_angles
            )
        
        return B_local
    
    def _build_topology_bnd_nodes(self):
        """Boundary nodes in topology vertex order (required by the C++ solid-angle kernel)."""
        topology_bnd_nodes = list(self.bnd_nodes[:self.global_n_bnd_nodes[self.rank]])

        # Match geometry ghosts to topology ghosts via shared global vertex id.
        for i_geom in range(
            self.domain.geometry.index_map().size_local,
            self.domain.geometry.index_map().size_local + 
            self.domain.geometry.index_map().num_ghosts
        ):
            for j_topo in range(
                self.domain.topology.index_map(0).size_local,
                self.domain.topology.index_map(0).size_local + 
                self.domain.topology.index_map(0).num_ghosts
            ):
                i_array = np.array([i_geom], dtype=np.int32)
                j_array = np.array([j_topo], dtype=np.int32)
                if (self.domain.geometry.index_map().local_to_global(i_array)[0] ==
                    self.domain.topology.index_map(0).local_to_global(j_array)[0]):
                    if i_geom in self.bnd_nodes:
                        topology_bnd_nodes.append(j_topo)
                    break
        
        return topology_bnd_nodes
    
    def _build_mesh_topology(self, topology_bnd_nodes):
        """Node/cell adjacency and coordinates for tetrahedron solid-angle quadrature."""
        self.domain.topology.create_connectivity(0, self.domain.topology.dim)
        n_to_c = self.domain.topology.connectivity(0, self.domain.topology.dim)
        
        self.domain.topology.create_connectivity(self.domain.topology.dim, 0)
        c_to_n = self.domain.topology.connectivity(self.domain.topology.dim, 0)
        
        node_to_cell = [
            list(n_to_c.links(i)) if n_to_c else []
            for i in range(
                self.domain.topology.index_map(0).size_local + 
                self.domain.topology.index_map(0).num_ghosts
            )
        ]
        
        cell_to_node = [
            list(c_to_n.links(i)) if c_to_n else []
            for i in range(self.domain.topology.index_map(self.domain.topology.dim).size_local)
        ]
        
        topology_nodes_coordinates = np.zeros((
            self.domain.topology.index_map(0).size_local + 
            self.domain.topology.index_map(0).num_ghosts, 3
        ))
        topology_nodes_coordinates[:self.domain.topology.index_map(0).size_local] = \
            self.domain.geometry.x[:self.domain.topology.index_map(0).size_local]
        
        for i_geom in range(
            self.domain.geometry.index_map().size_local,
            self.domain.geometry.index_map().size_local + 
            self.domain.geometry.index_map().num_ghosts
        ):
            for j_topo in range(
                self.domain.topology.index_map(0).size_local,
                self.domain.topology.index_map(0).size_local + 
                self.domain.topology.index_map(0).num_ghosts
            ):
                i_array = np.array([i_geom], dtype=np.int32)
                j_array = np.array([j_topo], dtype=np.int32)
                if (self.domain.geometry.index_map().local_to_global(i_array)[0] ==
                    self.domain.topology.index_map(0).local_to_global(j_array)[0]):
                    topology_nodes_coordinates[j_topo] = self.domain.geometry.x[i_geom]
                    break
        
        return {
            'node_to_cell': node_to_cell,
            'cell_to_node': cell_to_node,
            'topology_nodes_coordinates': topology_nodes_coordinates,
            'topology_bnd_nodes': np.array(topology_bnd_nodes, dtype=np.int32)
        }
    
    def save(self, filename):
        """
        Write ``BEM`` and indexing arrays to a binary file.

        Parameters
        ----------
        filename : str
            Output path; parent directory is created if needed.
        """
        if self.BEM is None:
            raise RuntimeError("BEM matrix not assembled. Call assemble() first.")
        
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            if self.verbose:
                print(f"  Created output directory: {output_dir}")
        
        n_local = self.global_n_bnd_nodes[self.rank]
        local_bnd_coords = self.domain.geometry.x[self.bnd_nodes[:n_local]]

        with open(filename, "wb") as f:
            np.save(f, self.BEM)
            np.save(f, self.global_n_bnd_nodes)
            np.save(f, self.gnodes_to_bnodes)
            np.save(f, self.bnd_nodes)
            np.save(f, local_bnd_coords)
        
        if self.verbose:
            print(f"✓ Saved BEM matrix to: {filename}")
    
    def load(self, filename):
        """
        Read a file written by :meth:`save`.

        Returns
        -------
        tuple
            ``(BEM, global_n_bnd_nodes, gnodes_to_bnodes, bnd_nodes, coordinates)``.
            ``coordinates`` is ``None`` in older files without that array.
        """
        with open(filename, "rb") as f:
            self.BEM = np.load(f)
            self.global_n_bnd_nodes = np.load(f)
            self.gnodes_to_bnodes = np.load(f)
            self.bnd_nodes = np.load(f)
            try:
                coordinates = np.load(f)
            except (EOFError, ValueError):
                coordinates = None
        
        if self.verbose:
            print(f"✓ Loaded BEM matrix from: {filename}")
        
        return self.BEM, self.global_n_bnd_nodes, self.gnodes_to_bnodes, self.bnd_nodes, coordinates

