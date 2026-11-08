cmake_minimum_required(VERSION 3.20)

# This build is for the NVHPC CUDA Fortran path. Set the compiler before
# project() so CMake detects the correct module format and dependency rules.
if(NOT DEFINED CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER nvfortran CACHE FILEPATH "Fortran compiler")
endif()

project(FDS LANGUAGES Fortran)

# Optional SUNDIALS dependency for CVODE detailed chemistry.
option(USE_SUNDIALS "Enable CVODE detailed chemistry through SUNDIALS" OFF)
option(SINGLE_PRECISION "Build with EB and QB in single precision" OFF)

# Source directory.
set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/Source")

# Store Fortran module files in the build directory.
set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/mod")

set(FDS_SOURCES
  ${SRC_DIR}/prec.f90
  ${SRC_DIR}/imkl.f90
  ${SRC_DIR}/prop.f90
  ${SRC_DIR}/cons.f90
  ${SRC_DIR}/pois.f90
  ${SRC_DIR}/devc.f90
  ${SRC_DIR}/type.f90
  ${SRC_DIR}/mesh.f90
  ${SRC_DIR}/func.f90
  ${SRC_DIR}/soot.f90
  ${SRC_DIR}/gsmv.f90
  ${SRC_DIR}/rcal.f90
  ${SRC_DIR}/ctrl.f90
  ${SRC_DIR}/turb.f90
  ${SRC_DIR}/data.f90
  ${SRC_DIR}/smvv.f90
  ${SRC_DIR}/geom.f90
  ${SRC_DIR}/vege.f90
  ${SRC_DIR}/init.f90
  ${SRC_DIR}/hvac.f90
  ${SRC_DIR}/ccib.f90
  ${SRC_DIR}/fire.f90
  ${SRC_DIR}/chem.f90
  ${SRC_DIR}/radi.f90
  ${SRC_DIR}/read.f90
  ${SRC_DIR}/dump.f90
  ${SRC_DIR}/pres.f90
  ${SRC_DIR}/wall.f90
  ${SRC_DIR}/velo.f90
  ${SRC_DIR}/part.f90
  ${SRC_DIR}/divg.f90
  ${SRC_DIR}/mass.f90
  ${SRC_DIR}/g_cons.f90
  ${SRC_DIR}/g_erro.f90
  ${SRC_DIR}/g_mesh.f90
  ${SRC_DIR}/g_ht1d.f90
  ${SRC_DIR}/g_dump.f90
  ${SRC_DIR}/g_init.f90
  ${SRC_DIR}/g_func.f90
  ${SRC_DIR}/g_exch.f90
  ${SRC_DIR}/g_fire.f90
  ${SRC_DIR}/g_pois.f90
  ${SRC_DIR}/g_mass.f90
  ${SRC_DIR}/g_wall.f90
  ${SRC_DIR}/g_divg.f90
  ${SRC_DIR}/g_pres.f90
  ${SRC_DIR}/g_velo.f90
  ${SRC_DIR}/g_radi.f90
  ${SRC_DIR}/g_tools.f90
  ${SRC_DIR}/main.f90
)

add_executable(fds ${FDS_SOURCES})
set_target_properties(fds PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
)
# NVHPC compiler options.
target_compile_options(fds PRIVATE
  -Mpreprocess
  -mp
  -cuda
  -O3

)

# Intel MKL interfaces.
target_compile_definitions(fds PRIVATE WITH_MKL)

# Revision in .out and .smv. Without .git (source copied as a tarball) pass -DGFDS_REVISION=... and
# -DGFDS_REVISION_DATE=...;
# otherwise "GFDS-" + git describe of the source tree.
set(GFDS_REVISION "" CACHE STRING "Revision string written to .out and .smv")
set(GFDS_REVISION_DATE "" CACHE STRING "Revision date for GFDS_REVISION (git log date if empty)")
if(NOT GFDS_REVISION)
  execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" describe --always --dirty --abbrev=10
                  OUTPUT_VARIABLE _gfds_rev OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _gfds_rc)
  if(_gfds_rc EQUAL 0 AND _gfds_rev)
    set(GFDS_REVISION "GFDS-${_gfds_rev}")
  endif()
endif()
if(GFDS_REVISION)
  set(_gfds_defs GITHASH_PP="${GFDS_REVISION}")
  set(_gfds_date "${GFDS_REVISION_DATE}")
  if(NOT _gfds_date)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" log -1 --format=%cd
                    OUTPUT_VARIABLE _gfds_date OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  if(_gfds_date)
    list(APPEND _gfds_defs GITDATE_PP="${_gfds_date}")
  endif()
endif()
string(TIMESTAMP _gfds_build "%b %d, %Y %H:%M:%S")
list(APPEND _gfds_defs BUILDDATE_PP="${_gfds_build}")
# only the two files that print them, so a new build date does not rebuild everything
set_source_files_properties(Source/func.f90 Source/dump.f90 PROPERTIES COMPILE_DEFINITIONS "${_gfds_defs}")

if(SINGLE_PRECISION)
  target_compile_definitions(fds PRIVATE GFDS_SINGLE_PRECISION)
endif()

if(USE_SUNDIALS)
  find_package(SUNDIALS CONFIG REQUIRED)
  target_link_libraries(fds PRIVATE SUNDIALS::fcvode_mod SUNDIALS::fnvecserial_mod)
  target_compile_definitions(fds PRIVATE WITH_SUNDIALS)
  if(DEFINED SUNDIALS_VERSION)
    target_compile_definitions(fds PRIVATE SUNDIALS_PP="${SUNDIALS_VERSION}")
  endif()
endif()

# NVHPC + OpenMPI path. Prefer an explicit -DNVHPC_MPI_ROOT or environment
# variable, then fall back to the OpenMPI bundled with the selected NVHPC SDK.
set(NVHPC_MPI_ROOT "$ENV{NVHPC_MPI_ROOT}" CACHE PATH
  "Root of the OpenMPI installation compatible with NVHPC")
if(NOT NVHPC_MPI_ROOT)
  if(DEFINED ENV{NVHPC_ROOT})
    set(_NVHPC_ROOT "$ENV{NVHPC_ROOT}")
  else()
    get_filename_component(_NVFORTRAN_BIN "${CMAKE_Fortran_COMPILER}" DIRECTORY)
    get_filename_component(_NVHPC_COMPILERS "${_NVFORTRAN_BIN}" DIRECTORY)
    get_filename_component(_NVHPC_ROOT "${_NVHPC_COMPILERS}" DIRECTORY)
  endif()
  file(GLOB _NVHPC_MPI_CANDIDATES LIST_DIRECTORIES true
    "${_NVHPC_ROOT}/comm_libs/*/openmpi*/openmpi-*")
  if(_NVHPC_MPI_CANDIDATES)
    list(GET _NVHPC_MPI_CANDIDATES 0 NVHPC_MPI_ROOT)
    set(NVHPC_MPI_ROOT "${NVHPC_MPI_ROOT}" CACHE PATH
      "Root of the OpenMPI installation compatible with NVHPC" FORCE)
  endif()
endif()
if(NOT EXISTS "${NVHPC_MPI_ROOT}/include" OR NOT EXISTS "${NVHPC_MPI_ROOT}/lib")
  message(FATAL_ERROR
    "Set NVHPC_MPI_ROOT to the compatible OpenMPI root containing include/ and lib/.\n"
    "Example: cmake -DNVHPC_MPI_ROOT=/path/to/openmpi ...")
endif()

target_include_directories(fds PRIVATE
  ${NVHPC_MPI_ROOT}/include
  ${NVHPC_MPI_ROOT}/lib
)

target_link_directories(fds PRIVATE ${NVHPC_MPI_ROOT}/lib)

# MPI Fortran libraries.
target_link_libraries(fds PRIVATE
  mpi_usempif08
  mpi_usempi_ignore_tkr
  mpi_mpifh
  mpi
)

# Runtime library paths relative to the executable.
set(RPATH_FLAGS "-Wl,-rpath,$ORIGIN:$ORIGIN/../../lib:$ORIGIN/../../../../../compilers/lib:$ORIGIN/../../../../../../compilers/lib")

# MKL cluster sparse solver (pressure solvers GLMAT, UGLMAT, ULMAT) loads its MPI wrappers from the BLACS
# library at run time ("Cannot load symbol MKLMPI_Get_wrappers" without it), so it must be linked even though
# nothing references it directly.
find_library(MKL_BLACS_LIB mkl_blacs_openmpi_lp64 HINTS ENV MKLROOT PATH_SUFFIXES lib lib/intel64)
if(MKL_BLACS_LIB)
  target_link_libraries(fds PRIVATE -Wl,--no-as-needed ${MKL_BLACS_LIB} -Wl,--as-needed)
else()
  message(WARNING "libmkl_blacs_openmpi_lp64 not found: GLMAT/UGLMAT/ULMAT pressure solvers will fail at run time")
endif()

target_link_options(fds PRIVATE
  ${RPATH_FLAGS}
  -Wl,-rpath,${NVHPC_MPI_ROOT}/lib
  -cuda
  -cudalib=nvtx
  -Mmkl
)
