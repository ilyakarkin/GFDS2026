# GFDS — FireCat GFDS fork

FireCat GFDS fork is a fork of [mxd45za/GFDS2026](https://github.com/mxd45za/GFDS2026) maintained for fire-safety
engineering models: smoke control and fire-risk studies of buildings (many rooms and meshes, timed and spreading
fires, detectors, sprinklers, fans, leaks). The goal is GPU runs that are **several times faster than FDS 6.11.1 on
CPU and give the same results**.

Author and maintainer of the fork: **Ilya Karkin**, [pyrosim.ru](https://pyrosim.ru).

The default branch `firesim` holds the current release. The releases are tagged (`firecat-rc9b`, …); the notes
of each release list what was checked.

## Release firecat-rc9b

Since November 2026 this build runs the GPU jobs of the FireSim remote FDS service. A job goes to the GPU only if
GFDS accepts the model; otherwise the same input runs on FDS 6.11.1 on CPU.

### What runs on the GPU

- fires: `HRRPUA` with `SPREAD_RATE` and `RAMP`, `BURN_AWAY`, `MATL` with `RAMP` properties (no pyrolysis reactions);
- lumped species, soot, `VISIBILITY` and other gas quantities; `SLCF`, `BNDF`, `DEVC`, `ISOF`, `SMOKE3D`;
- `OBST`/`HOLE` (also created or removed by devices), many meshes, one or several MPI processes per GPU;
- `VENT` with `RAMP`, `DEVC`/`CTRL`, pressure zones with `LEAK`, `CONSTANT_SPECIFIC_HEAT_RATIO`;
- wind, Deardorff turbulence, FFT pressure solver (`GLMAT` too);
- Lagrangian particles: sprinklers, water and fuel sprays (on the host, coupled to the GPU gas phase).

### What is refused on the GPU

GFDS stops at set-up with one standard line, and the model should be run on CPU (`--c` or FDS):

    ERROR: GPU: <what> is not supported; run this case with --c

This covers solid-phase pyrolysis (`MATL` with reactions), `THERMOCOUPLE`, cylindrical geometry, solid
particles, CVODE chemistry, `BLOWING`, `MP5`, turbulence models other than Deardorff, and output quantities that have
no GPU field yet (for example `PARTICLE FLUX X/Y/Z`, which differed from FDS 6.11.1 by 10–20 %).

### How it was checked

Three runs of every case: g — GFDS on GPU, c — GFDS on CPU (`--c`), f — FDS 6.11.1. A case passes if g−f is not
larger than c−f, or is within the sensitivity of FDS itself (f against f with `TMPA` changed by 1e-6 K), or within
the run-to-run scatter of the GPU.

- FDS verification cases (299): 230 run on the GPU and pass; 69 are refused in the standard form (pyrolysis 29,
  set-up only / TGA 12, cylindrical 7, CVODE 6, thermocouples 5, solid particles 5) or are not plain runs
  (restart parts, a case that FDS 6.11.1 does not read either, one very long radiation case left for later).
- Building models of testers (smoke control and fire risk), first 120 s: within the sensitivity of FDS 6.11.1.
- Speed on an NVIDIA A100 against FDS 6.11.1 on CPU: 4–19 times faster on the testers' models, about 8 times
  typically; a 2.7 M-cell model (850 s) took 57 minutes, against about 50 hours estimated on CPU.

### Output

Same files as FDS 6.11.1. The `.out` and `.smv` revision line is `FDS-6.11.1 (GPU)`: the results are checked
against FDS 6.11.1; the next line of `.out`, `Computed on GPU : GFDS-FireCat-<revision>`, names the actual build.
`CHID_MODEL_TIME.csv` (time per module) is written in addition.

GFDS is based on the FDS development version of 6 May 2025 (`66384f83df`), not on FDS 6.11.1; the FDS 6.11.1
changes that matter for building models are ported here (equation of state at cell faces, gas absorption look-up,
output formats, the random sequence and others).

### Building and running

Build with NVHPC (`nvfortran`, CUDA 12.x, OpenMPI from the NVHPC SDK) and MKL:

    cmake -B build -DCMAKE_BUILD_TYPE=Release -DNVHPC_MPI_ROOT=<openmpi root> -DCMAKE_Fortran_FLAGS='-gpu=cc80'
    make -C build -j8

To share one GPU between several jobs, start the CUDA MPS daemon (`nvidia-cuda-mps-control -d`); multi-mesh cases
run faster with one MPI process per mesh.

---

# GFDS

Version: 1.0

GFDS is a CUDA Fortran branch of the Fire Dynamics Simulator (FDS) for NVIDIA GPUs. It calculates three-dimensional low-Mach-number flow, multi-species transport, combustion, thermal radiation, and wall heat transfer. It also supports multi-mesh calculations on one or more GPUs.

Input files and configuration examples are provided in [`case/`](case/).

## Build

GFDS requires the NVIDIA HPC SDK with `nvfortran`, CUDA, an OpenMPI installation compatible with NVHPC, CMake 3.20 or newer, and GNU Make.

Double precision is the default:

```bash
cmake -S . -B build
cmake --build build -j8
```

To specify the OpenMPI installation directory explicitly:

```bash
cmake -S . -B build -DNVHPC_MPI_ROOT=/path/to/openmpi
cmake --build build -j8
```

For single precision, use a separate build directory:

```bash
cmake -S . -B build-single -DSINGLE_PRECISION=ON
cmake --build build-single -j8
```

The resulting executable is `build/fds`.

## Run

Run a case in GPU mode:

```bash
mpirun -np 1 build/fds "case/case-3-Billaud/RA6.fds" --g
```

The execution modes are:

| Option | Mode |
| --- | --- |
| `--g` or `--gpu` | GPU mode and the default execution mode |
| `--gc` | GFDS call path using the corresponding CPU numerical functions for comparison and module timing |
| `--c` or `--cpu` | Original FDS CPU time-step path |

A one-to-one mapping between MPI processes and GPUs is recommended. For example, run a four-mesh case with four MPI processes and four GPUs:

```bash
mpirun -np 4 build/fds "case/case-7-Speedup/Arup Tunnel/4MESH/T_100_4.fds" --g
```

Fewer MPI processes can also run a multi-mesh case when fewer GPUs are available. For example, one MPI process can run all four meshes on one GPU:

```bash
mpirun -np 1 build/fds "case/case-7-Speedup/Arup Tunnel/4MESH/T_100_4.fds" --g
```

## Python Plotting

[`GUtilities/slice_contour.py`](GUtilities/slice_contour.py) extracts FDS slice fields and generates contour or vector figures together with the plotted CSV data. Installation and example commands are provided in the [Python plotting guide](GUtilities/USAGE.md). The complete command reference is available from:

```bash
python GUtilities/slice_contour.py --help
```

## Contact

- [xiaoxm8@mail2.sysu.edu.cn](mailto:xiaoxm8@mail2.sysu.edu.cn)
- [mingpj@mail.sysu.edu.cn](mailto:mingpj@mail.sysu.edu.cn)

## Citation and License

Citation metadata are provided in [CITATION.cff](CITATION.cff). The original FDS/NIST license notice is retained in [LICENSE.md](LICENSE.md), and the GFDS modification notice is provided in [NOTICE.md](NOTICE.md).

GFDS makes no warranty of any kind. Users assume all risks arising from its use.

## Acknowledgment

Thanks to Marcos Vanella for feedback from testing GFDS on different GPU platforms.
