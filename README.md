# GFDS — FireCat GFDS fork

FireCat GFDS fork is a fork of [mxd45za/GFDS2026](https://github.com/mxd45za/GFDS2026) maintained for fire-safety
engineering models (building evacuation studies: many rooms, timed burners, smoke and heat detectors).
The goal is GPU runs **5–10 times faster than FDS 6.11.1 on CPU with the same results**.

Author and maintainer of the fork: **Ilya Karkin**, [pyrosim.ru](https://pyrosim.ru).

The default branch `firesim` is upstream `GFDS` plus the changes below. Each change is also a separate
branch and a pull request to upstream, so anyone can take only what they need. Upstream changes are merged
in from time to time.

| Branch | Change | Upstream PR |
|---|---|---|
| `gpu-detector-fields` | Smoke and heat detectors (`CHAMBER OBSCURATION`, `PATH OBSCURATION`, `TRANSMISSION`, `LINK TEMPERATURE`) no longer stop a `--g` run | [#1](https://github.com/mxd45za/GFDS2026/pull/1) |
| `gpu-radiative-heat-flux-gas` | `RADIATIVE HEAT FLUX GAS` devices with `ORIENTATION` report the flux on GPU (were 0) | [#2](https://github.com/mxd45za/GFDS2026/pull/2) |
| `gpu-vent-activation` | `VENT`s with `DEVC_ID`/`CTRL_ID` switch on and off on GPU; `OBST`/`HOLE` created or removed during a run stop it with a message instead of being ignored | [#3](https://github.com/mxd45za/GFDS2026/pull/3) |
| `gpu-zone-perf` | Models with many pressure zones: 1.8x faster time stepping (165 zones) | [#4](https://github.com/mxd45za/GFDS2026/pull/4) |
| `gpu-radiation-sweep` | Faster radiation sweep, identical results (6–11 % faster runs) | [#5](https://github.com/mxd45za/GFDS2026/pull/5) |

Every change is checked against FDS 6.11.1 on CPU, on FDS verification cases and on building models; the
details are in the pull requests.

### Known differences from FDS 6.11.1

GFDS is based on the FDS development version of 6 May 2025 (`66384f83df`), not on FDS 6.11.1. Both the CPU
path (`--c`) and the GPU path inherit the differences, among them:

- the gas absorption coefficient lookup (fixed in FDS by `0eab2e6d17`): ambient air gets 0.109 1/m instead
  of 0.085 1/m;
- `FLUX_LIMITER_MW_CORRECTION` is off unless `SIMULATION_MODE='LES'` (on by default in FDS 6.11.1) and is
  not implemented in the GPU kernels. In models with many closed rooms this lets the background pressure of
  the closed zones drift.

Porting these fixes is the current work in this fork.

### Output

Same files as FDS, plus `CHID_MODEL_TIME.csv` (time per module). Differences: device names in `_devc.csv`
are quoted, the third Smoke3D channel is labelled `TEMPERATURE`, `.out` shows `Revision: unknown`.

### Running several jobs on one GPU

Start the CUDA MPS daemon (`nvidia-cuda-mps-control -d`) and run multi-mesh cases with one MPI process per
mesh: on an A100 a 3-mesh case ran 1.7x faster than with one process, and 3–4 jobs can share one card.

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
