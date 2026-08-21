"""MPI-distributed demagnetizing BEM matrix assembly (``bem_cpp`` / ``bem_cpp_debug``)."""

from bem_assembler.bem_assembler import (
    BEMAssembler,
    configure_cpp_backend,
    compute_solid_angles,
    format_bytes,
    format_bytes_range,
    get_cpp_backend,
    get_cpp_backend_kind,
    is_cpp_available,
    load_cpp_backend,
)

__all__ = [
    "BEMAssembler",
    "configure_cpp_backend",
    "compute_solid_angles",
    "format_bytes",
    "format_bytes_range",
    "get_cpp_backend",
    "get_cpp_backend_kind",
    "is_cpp_available",
    "load_cpp_backend",
]
