# GFDS — FireCat GFDS fork

FireCat GFDS fork is a fork of [mxd45za/GFDS2026](https://github.com/mxd45za/GFDS2026) (GFDS, a GPU port of
[NIST FDS](https://github.com/firemodels/fds)), maintained for fire-safety engineering models (building evacuation
studies: many rooms, timed burners, smoke and heat detectors).
The goal is GPU runs **5–10 times faster than FDS 6.11.1 on CPU with the same results**. On building fire models on
an NVIDIA A100 this release is 4–19 times faster than FDS 6.11.1 on CPU, typically about 8 times.

Author and maintainer of the fork: **Ilya Karkin**, [pyrosim.ru](https://pyrosim.ru).

The default branch `firesim` holds the current release, `firecat-611`: the GFDS GPU code on the FDS 6.11.1 source
(tag `FDS-6.11.1`).

### What runs on the GPU

- fires: `HRRPUA` with `SPREAD_RATE`, `BURN_AWAY`, `MATL` with `RAMP` properties;
- lumped species and `VISIBILITY`; `SLCF`, `BNDF`, `DEVC`, including smoke and heat detectors;
- `OBST`/`HOLE`, many meshes, `VENT` with `RAMP`, `DEVC`/`CTRL`, pressure zones with `LEAK`;
- wind, Deardorff turbulence, the FFT pressure solver, `CONSTANT_SPECIFIC_HEAT_RATIO`;
- particles and sprinklers (on the host).

### How it was checked

FDS verification suite, 306 cases: 230 run on the GPU; the others do not (explicit refusals that go to CPU,
cases that only do set-up or TGA, input errors and similar). 293 cases were compared with FDS 6.11.1. Above the
comparison threshold:

- `spreading_fire`: the burning area in the decay after 180 s (HRR within 0.6 %);
- `thick_orifice_5cm`, `radiation_gas_panel`: within the sensitivity of FDS 6.11.1 itself;
- `box_burn_away3` (**known open difference**): two `hrr` rows at the moment an obstruction burns away differ from
  FDS 6.11.1 on the GPU (`Q_CONV` 167 against 215 kW); the CPU path equals FDS 6.11.1. The previous version
  behaves the same; not fixed.

Building models, 850 s: mean HRR within 0.1–0.3 %. The pressure in the fire zone still diverges, as in the
previous version; check models where the zone pressure matters against a CPU run.

### Differences from FDS 6.11.1

The CPU source is FDS 6.11.1 (tag `FDS-6.11.1`); `--c` runs the FDS 6.11.1 time loop. The source differs from
6.11.1 only by:

- **GPU hooks:** the GPU time loop and the GPU feature check in `main.f90`, `PUBLIC` lines and a few fields that
  the GPU code reads, and the revision strings (`Revision : FDS-6.11.1 (GPU)`, `Computed on GPU : GFDS-FireCat-…`).
- **nvfortran compatibility:** plain `IMPLICIT NONE` (nvfortran rejects `IMPLICIT NONE (TYPE,EXTERNAL)`), the
  random numbers of the 6.11.1 release (`FDS_RANDOM`), input files read through a normalized copy that parses as
  with Intel Fortran, arrays freed before `MOVE_ALLOC` (memory leak), one type declared before its use,
  `RDXINT=N*(1/L)` as the release build evaluates it, and the limiting species set to zero in fast chemistry
  (the release build, without FMA, leaves zero there).
- **Host-side speed-ups of the GPU path.** The same code runs on the CPU path and leaves the results unchanged:
  all output files are byte for byte equal to the build without them, on the CPU path, and on the GPU path except
  where two runs of the same build already differ (last digits of a few values):

| Area | Change | Why it is not FDS 6.11.1 |
|---|---|---|
| Pressure zones: map of thin OBST faces | `ASSIGN_PRESSURE_ZONE` marks thin-OBST faces once per mesh and keeps its queue | 6.11.1 loops over all OBSTs at every face; set-up of many zones took minutes |
| Pressure zones: search only meshes next to a change | `ZONE_BOUNDARY_EXCHANGE` revisits only meshes that can change | set-up cost no longer grows as zones × meshes |
| Set-up: grow storage geometrically | `ALLOCATE_STORAGE` grows WALL/BOUNDARY arrays by 25 % instead of 1000 slots; the `.smv` string doubles | 6.11.1 copies the arrays O(N²/1000) times; set-up of models with thousands of OBSTs took minutes |
| Slices: `.sf.bnd` bounds in memory | the slice bounds file is read once per run | 6.11.1 re-opens the file at every slice dump |
| Particles: skip vent insertion loops | `INSERT_VENT_PARTICLES` returns at once when no SURF inserts particles | 6.11.1 visits every wall cell every step for nothing |
| Slices: keep the cell weights | `DUMP_SLCF` keeps its gas/solid weights until the geometry changes | 6.11.1 recomputes them (with a mesh search per exterior cell) at every dump |
| Gas-phase output: no heap allocation | `GAS_PHASE_OUTPUT` work arrays allocated once | nvfortran puts its automatic arrays on the heap at every call (every slice cell) |
| Devices: compare SURF_ID once per run | surface statistics compare the SURF_ID string once per run of equal SURF_INDEX | 6.11.1 compares it for every wall cell every step |
| Devices: SPATIAL_STATISTIC as an integer | gas-phase spatial statistics map the statistic to an integer and look up the position of the output quantity once per device | 6.11.1 compares the statistic string for every cell every step; a 6-mesh building with two whole-floor VOLUME MEAN devices spent 40 % of its GPU step there |
| Devices and slices: plain cell values | DENSITY, TEMPERATURE, U/V/W-VELOCITY, MASS FRACTION, VISIBILITY of device statistics and U/V/W-VELOCITY, VELOCITY of slices are evaluated in the cell loop with the expressions of `GAS_PHASE_OUTPUT` (no cut cells, no DRY) | 6.11.1 calls `GAS_PHASE_OUTPUT` for every cell; this is a second copy of those expressions, to be kept equal to it |

Features that the GPU time loop does not implement stop a `--g` run at set-up with
`ERROR: GPU: … is not supported; run this case with --c`, and the case has to be run on CPU: pyrolysis (`MATL` with
reactions), `THERMOCOUPLE`, cylindrical geometry, solid particles, CVODE, `BLOWING`, `MP5`, turbulence models other
than Deardorff, complex geometry (`GEOM`, CC_IBM), a burning thin obstruction inside a mesh, LES/DNS at the boundary
of meshes of different cell size, non-standard `HEAT_TRANSFER_MODEL`, and output quantities with no GPU mapping.

The background pressure of closed rooms is very sensitive (1 Pa is about 0.003 K of the mean room
temperature) and still differs from FDS 6.11.1 by a few pascals.

**Known GPU issue (under investigation):** in a model with 165 pressure zones, closed rooms next to the fire
room sometimes become unstable on GPU after 130–360 s: temperature swings by tens of degrees (down to
negative °C) and the zone pressure runs away. FDS 6.11.1 and the GFDS CPU path do not show it. Models with
many closed rooms should be checked against a CPU run until this is fixed.

### Output

Same files as FDS 6.11.1, plus `CHID_MODEL_TIME.csv` (time per module) in `--g` runs when
`GFDS_MODEL_TIME` is set. The `.out` of a `--g` run
shows `Revision : FDS-6.11.1 (GPU)` and the GFDS build in the `Computed on GPU` line.

### Running several jobs on one GPU

Start the CUDA MPS daemon (`nvidia-cuda-mps-control -d`) and run multi-mesh cases with one MPI process per
mesh: on an A100 a 3-mesh case ran 1.7x faster than with one process, and 3–4 jobs can share one card.

---

# GFDS

Upstream GFDS version: 1.0.

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

## Citation and License

Citation metadata are provided in [CITATION.cff](CITATION.cff). The original FDS/NIST license notice is retained in [LICENSE.md](LICENSE.md), and the GFDS modification notice is provided in [NOTICE.md](NOTICE.md).

GFDS makes no warranty of any kind. Users assume all risks arising from its use.

## Acknowledgment

Thanks to Marcos Vanella for feedback from testing GFDS on different GPU platforms.
