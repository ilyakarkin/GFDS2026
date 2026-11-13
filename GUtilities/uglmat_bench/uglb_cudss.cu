// Offline bench of the UGLMAT pressure matrices of GFDS with NVIDIA cuDSS.
//
// Input: the files written by a GFDS bench build with GFDS_UGL_DUMP=<dir> (pres.f90, UGLB_WRITE_ZONE):
//   <dir>/<chid>_z<Z>_mat.bin      int32 n, nnz, mkl_mtype, call; int32 ia[n+1], ja[nnz] (1-based, upper triangle, CSR);
//                                  real64 a[nnz]
//   <dir>/<chid>_z<Z>_c<C>_rhs.bin int32 n; real64 f[n]  (RHS given to MKL cluster_sparse_solver)
//   <dir>/<chid>_z<Z>_c<C>_sol.bin int32 n; real64 x[n]  (its solution)
//
// Usage: uglb_cudss <dir or lap:NX,NY,NZ> <chid> "<zones>" "<calls>" <spd|sym|gen> <sep|blk> [reps] [pin=1]
//   sep — one cuDSS system per zone, solved one after another (as UGLMAT does now);
//   blk — all the listed zones in one block-diagonal system, one solve per call.
//   pin — a singular zone (mkl_mtype -2, no Dirichlet boundary) gets its last unknown fixed to 0, which makes it SPD.
//
// Printed per system: n, nnz (upper), analysis, factorization, solve (data on the device), solve with H2D of the RHS
// and D2H of the solution (pinned and pageable host memory), device memory, relative difference to MKL, residuals.

#include <cuda_runtime.h>
#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", e_, cudaGetErrorString(e_), __FILE__, \
                    __LINE__);                                                                   \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)
#define CS(x)                                                                                    \
    do {                                                                                         \
        cudssStatus_t s_ = (x);                                                                  \
        if (s_ != CUDSS_STATUS_SUCCESS) {                                                        \
            fprintf(stderr, "cuDSS status %d at %s:%d\n", (int)s_, __FILE__, __LINE__);          \
            exit(3);                                                                             \
        }                                                                                        \
    } while (0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Zone {
    int id = 0, n = 0, nnz = 0, mtype = 0;
    std::vector<int> ia, ja;  // 0-based after reading
    std::vector<double> a;
    std::vector<std::vector<double>> rhs, sol;
    std::vector<int> calls;
};

// Synthetic zone for scaling: 7-point Laplacian on an NX x NY x NZ box, Dirichlet on the top face (SPD), Neumann
// elsewhere; RHS = sin pattern; the "MKL solution" is left at zero (only the residual is meaningful).
static void lap_zone(int nx, int ny, int nz, int ncalls, Zone &Z) {
    Z.id = 0; Z.n = nx * ny * nz; Z.mtype = 2;
    auto id = [&](int i, int j, int k) { return i + nx * (j + ny * k); };
    Z.ia.assign(1, 0);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                double d = (i > 0) + (i < nx - 1) + (j > 0) + (j < ny - 1) + (k > 0) + (k < nz - 1) + (k == nz - 1 ? 2 : 0);
                Z.ja.push_back(id(i, j, k)); Z.a.push_back(d);
                if (i < nx - 1) { Z.ja.push_back(id(i + 1, j, k)); Z.a.push_back(-1); }
                if (j < ny - 1) { Z.ja.push_back(id(i, j + 1, k)); Z.a.push_back(-1); }
                if (k < nz - 1) { Z.ja.push_back(id(i, j, k + 1)); Z.a.push_back(-1); }
                Z.ia.push_back(Z.ja.size());
            }
    Z.nnz = Z.ja.size();
    for (int c = 0; c < ncalls; c++) {
        std::vector<double> b(Z.n), x(Z.n, 0.0);
        for (int q = 0; q < Z.n; q++) b[q] = sin(0.001 * q * (c + 1));
        Z.rhs.push_back(b); Z.sol.push_back(x); Z.calls.push_back(c);
    }
}

static bool read_zone(const std::string &dir, const std::string &chid, int z, const std::vector<int> &calls,
                      Zone &Z) {
    if (dir.rfind("lap:", 0) == 0) {
        if (z != 0) return false;
        int nx, ny, nz;
        if (sscanf(dir.c_str() + 4, "%d,%d,%d", &nx, &ny, &nz) != 3) exit(6);
        lap_zone(nx, ny, nz, calls.size(), Z);
        return true;
    }
    std::string fn = dir + "/" + chid + "_z" + std::to_string(z) + "_mat.bin";
    FILE *f = fopen(fn.c_str(), "rb");
    if (!f) return false;
    int h[4];
    if (fread(h, 4, 4, f) != 4) exit(4);
    Z.id = z; Z.n = h[0]; Z.nnz = h[1]; Z.mtype = h[2];
    Z.ia.resize(Z.n + 1); Z.ja.resize(Z.nnz); Z.a.resize(Z.nnz);
    if (fread(Z.ia.data(), 4, Z.n + 1, f) != (size_t)Z.n + 1) exit(4);
    if (fread(Z.ja.data(), 4, Z.nnz, f) != (size_t)Z.nnz) exit(4);
    if (fread(Z.a.data(), 8, Z.nnz, f) != (size_t)Z.nnz) exit(4);
    fclose(f);
    for (auto &v : Z.ia) v -= 1;
    for (auto &v : Z.ja) v -= 1;
    // sort the columns of every row
    for (int i = 0; i < Z.n; i++) {
        std::vector<std::pair<int, double>> r;
        for (int k = Z.ia[i]; k < Z.ia[i + 1]; k++) r.push_back({Z.ja[k], Z.a[k]});
        std::sort(r.begin(), r.end());
        for (int k = Z.ia[i]; k < Z.ia[i + 1]; k++) { Z.ja[k] = r[k - Z.ia[i]].first; Z.a[k] = r[k - Z.ia[i]].second; }
    }
    for (int c : calls) {
        std::vector<double> v[2];
        const char *w[2] = {"rhs", "sol"};
        bool ok = true;
        for (int j = 0; j < 2; j++) {
            std::string fv = dir + "/" + chid + "_z" + std::to_string(z) + "_c" + std::to_string(c) + "_" + w[j] + ".bin";
            FILE *g = fopen(fv.c_str(), "rb");
            if (!g) { ok = false; break; }
            int n;
            if (fread(&n, 4, 1, g) != 1 || n != Z.n) exit(5);
            v[j].resize(n);
            if (fread(v[j].data(), 8, n, g) != (size_t)n) exit(5);
            fclose(g);
        }
        if (!ok) continue;
        Z.rhs.push_back(v[0]); Z.sol.push_back(v[1]); Z.calls.push_back(c);
    }
    return true;
}

// y = A x with A symmetric given by its upper triangle
static void symv(const Zone &Z, const double *x, double *y) {
    for (int i = 0; i < Z.n; i++) y[i] = 0;
    for (int i = 0; i < Z.n; i++)
        for (int k = Z.ia[i]; k < Z.ia[i + 1]; k++) {
            int j = Z.ja[k];
            y[i] += Z.a[k] * x[j];
            if (j != i) y[j] += Z.a[k] * x[i];
        }
}
static double nrm(const double *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += x[i] * x[i]; return sqrt(s); }
static double rel_res(const Zone &Z, const double *b, const double *x) {
    std::vector<double> r(Z.n);
    symv(Z, x, r.data());
    for (int i = 0; i < Z.n; i++) r[i] = b[i] - r[i];
    return nrm(r.data(), Z.n) / std::max(nrm(b, Z.n), 1e-300);
}
static void demean(std::vector<double> &x) { double m = 0; for (double v : x) m += v; m /= x.size(); for (double &v : x) v -= m; }

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: see the head of uglb_cudss.cu\n"); return 1; }
    std::string dir = argv[1], chid = argv[2], mode = argv[5], layout = argv[6];
    int reps = argc > 7 ? atoi(argv[7]) : 30;
    int pin = argc > 8 ? atoi(argv[8]) : 1;
    std::vector<int> zl, cl;
    { std::stringstream s(argv[3]); int v; while (s >> v) zl.push_back(v); }
    { std::stringstream s(argv[4]); int v; while (s >> v) cl.push_back(v); }

    std::vector<Zone> zones;
    for (int z : zl) { Zone Z; if (read_zone(dir, chid, z, cl, Z)) zones.push_back(Z); }
    if (zones.empty()) { fprintf(stderr, "no zones read\n"); return 1; }
    size_t nr = zones[0].rhs.size();
    for (auto &Z : zones) nr = std::min(nr, Z.rhs.size());
    printf("# cuDSS %d.%d.%d, mode=%s layout=%s reps=%d pin=%d, right-hand sides per zone: %zu\n", CUDSS_VERSION_MAJOR,
           CUDSS_VERSION_MINOR, CUDSS_VERSION_PATCH, mode.c_str(), layout.c_str(), reps, pin, nr);

    // the systems: one per zone, or one block-diagonal of all zones
    std::vector<std::vector<int>> systems;
    if (layout == "blk") { std::vector<int> s; for (size_t i = 0; i < zones.size(); i++) s.push_back(i); systems.push_back(s); }
    else for (size_t i = 0; i < zones.size(); i++) systems.push_back({(int)i});

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudssHandle_t handle; CS(cudssCreate(&handle)); CS(cudssSetStream(handle, stream));

    double sum_solve = 0, sum_solve_pin = 0, sum_solve_page = 0, sum_an = 0, sum_fa = 0;
    for (auto &sys : systems) {
        // assemble the (block-diagonal) matrix: upper (spd/sym) or full (gen), 0-based
        int n = 0;
        std::vector<int> off;
        for (int iz : sys) { off.push_back(n); n += zones[iz].n; }
        std::vector<std::vector<std::pair<int, double>>> rows(n);
        std::vector<double> sgnz(sys.size(), 1.0);
        for (size_t q = 0; q < sys.size(); q++) {
            const Zone &Z = zones[sys[q]];
            double d0 = Z.a[Z.ia[0]];  // the first entry of row 0 is its diagonal (columns sorted, upper)
            double sgn = d0 < 0 ? -1 : 1;  // FDS may store -A; cuDSS SPD needs +A
            sgnz[q] = sgn;
            int pinrow = (pin && Z.mtype == -2) ? Z.n - 1 : -1;
            for (int i = 0; i < Z.n; i++)
                for (int k = Z.ia[i]; k < Z.ia[i + 1]; k++) {
                    int j = Z.ja[k];
                    double v = sgn * Z.a[k];
                    if (pinrow >= 0 && (i == pinrow || j == pinrow)) v = (i == j) ? 1.0 : 0.0;
                    if (pinrow >= 0 && (i == pinrow || j == pinrow) && i != j) continue;
                    rows[off[q] + i].push_back({off[q] + j, v});
                    if (mode == "gen" && j != i) rows[off[q] + j].push_back({off[q] + i, v});
                }
        }
        std::vector<int> ia(n + 1, 0), ja;
        std::vector<double> a;
        for (int i = 0; i < n; i++) {
            std::sort(rows[i].begin(), rows[i].end());
            for (auto &p : rows[i]) { ja.push_back(p.first); a.push_back(p.second); }
            ia[i + 1] = ja.size();
        }
        int nnz = ja.size();
        rows.clear(); rows.shrink_to_fit();

        size_t free0, tot; CK(cudaMemGetInfo(&free0, &tot));
        int *d_ia, *d_ja; double *d_a, *d_b, *d_x;
        CK(cudaMalloc(&d_ia, (n + 1) * 4)); CK(cudaMalloc(&d_ja, (size_t)nnz * 4)); CK(cudaMalloc(&d_a, (size_t)nnz * 8));
        CK(cudaMalloc(&d_b, (size_t)n * 8)); CK(cudaMalloc(&d_x, (size_t)n * 8));
        CK(cudaMemcpy(d_ia, ia.data(), (n + 1) * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ja, ja.data(), (size_t)nnz * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_a, a.data(), (size_t)nnz * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_b, 0, (size_t)n * 8)); CK(cudaMemset(d_x, 0, (size_t)n * 8));

        cudssConfig_t cfg; cudssData_t data;
        CS(cudssConfigCreate(&cfg)); CS(cudssDataCreate(handle, &data));
        cudssMatrix_t A, B, X;
        cudssMatrixType_t mt = mode == "spd" ? CUDSS_MTYPE_SPD : mode == "sym" ? CUDSS_MTYPE_SYMMETRIC : CUDSS_MTYPE_GENERAL;
        cudssMatrixViewType_t mv = mode == "gen" ? CUDSS_MVIEW_FULL : CUDSS_MVIEW_UPPER;
        CS(cudssMatrixCreateCsr(&A, n, n, nnz, d_ia, NULL, d_ja, d_a, CUDA_R_32I, CUDA_R_32I, CUDA_R_64F, mt, mv,
                                CUDSS_BASE_ZERO));
        CS(cudssMatrixCreateDn(&B, n, 1, n, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CS(cudssMatrixCreateDn(&X, n, 1, n, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

        double t0 = now();
        CS(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, cfg, data, A, X, B)); CK(cudaStreamSynchronize(stream));
        double t1 = now();
        CS(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, cfg, data, A, X, B)); CK(cudaStreamSynchronize(stream));
        double t2 = now();
        int info = 0; size_t sz;
        CS(cudssDataGet(handle, data, CUDSS_DATA_INFO, &info, sizeof(info), &sz));
        int64_t lunnz = 0; CS(cudssDataGet(handle, data, CUDSS_DATA_LU_NNZ, &lunnz, sizeof(lunnz), &sz));
        int64_t mem[16] = {0}; CS(cudssDataGet(handle, data, CUDSS_DATA_MEMORY_ESTIMATES, mem, sizeof(mem), &sz));
        int npiv = 0; cudssDataGet(handle, data, CUDSS_DATA_NPIVOTS, &npiv, sizeof(npiv), &sz);
        // a second factorization with the same values (what an OBST change costs besides the analysis)
        double t3 = now();
        CS(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, cfg, data, A, X, B)); CK(cudaStreamSynchronize(stream));
        double t4 = now();
        size_t free1; CK(cudaMemGetInfo(&free1, &tot));

        // host buffers: pinned and pageable
        double *hb_pin, *hx_pin; CK(cudaMallocHost(&hb_pin, (size_t)n * 8)); CK(cudaMallocHost(&hx_pin, (size_t)n * 8));
        std::vector<double> hb(n), hx(n);

        std::vector<double> ts_dev, ts_pin, ts_page;
        double worst_rel = 0, worst_res = 0, worst_res_mkl = 0;
        for (size_t r = 0; r < nr; r++) {
            for (size_t q = 0; q < sys.size(); q++) {
                const Zone &Z = zones[sys[q]];
                for (int i = 0; i < Z.n; i++) hb[off[q] + i] = sgnz[q] * Z.rhs[r][i];
                if (pin && Z.mtype == -2) hb[off[q] + Z.n - 1] = 0;
            }
            memcpy(hb_pin, hb.data(), (size_t)n * 8);
            CK(cudaMemcpy(d_b, hb.data(), (size_t)n * 8, cudaMemcpyHostToDevice));
            for (int k = 0; k < reps; k++) {  // data on the device
                double s0 = now();
                CS(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, X, B)); CK(cudaStreamSynchronize(stream));
                ts_dev.push_back(now() - s0);
            }
            for (int k = 0; k < reps; k++) {  // pinned host RHS/solution
                double s0 = now();
                CK(cudaMemcpyAsync(d_b, hb_pin, (size_t)n * 8, cudaMemcpyHostToDevice, stream));
                CS(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, X, B));
                CK(cudaMemcpyAsync(hx_pin, d_x, (size_t)n * 8, cudaMemcpyDeviceToHost, stream));
                CK(cudaStreamSynchronize(stream));
                ts_pin.push_back(now() - s0);
            }
            for (int k = 0; k < reps; k++) {  // pageable host RHS/solution
                double s0 = now();
                CK(cudaMemcpy(d_b, hb.data(), (size_t)n * 8, cudaMemcpyHostToDevice));
                CS(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, X, B));
                CK(cudaMemcpy(hx.data(), d_x, (size_t)n * 8, cudaMemcpyDeviceToHost));
                ts_page.push_back(now() - s0);
            }
            // accuracy per zone against MKL
            for (size_t q = 0; q < sys.size(); q++) {
                const Zone &Z = zones[sys[q]];
                std::vector<double> xc(hx.begin() + off[q], hx.begin() + off[q] + Z.n), xm = Z.sol[r];
                double rc = rel_res(Z, Z.rhs[r].data(), xc.data()), rm = rel_res(Z, Z.rhs[r].data(), xm.data());
                if (Z.mtype == -2) { demean(xc); demean(xm); }
                std::vector<double> d(Z.n); for (int i = 0; i < Z.n; i++) d[i] = xc[i] - xm[i];
                double rel = nrm(d.data(), Z.n) / std::max(nrm(xm.data(), Z.n), 1e-300);
                printf("  zone %d call %d: n=%d mkl_mtype=%d |x_cudss-x_mkl|/|x_mkl|=%.2e res_cudss=%.2e res_mkl=%.2e\n",
                       Z.id, Z.calls[r], Z.n, Z.mtype, rel, rc, rm);
                worst_rel = std::max(worst_rel, rel); worst_res = std::max(worst_res, rc); worst_res_mkl = std::max(worst_res_mkl, rm);
            }
        }
        double md = median(ts_dev) * 1e3, mp = median(ts_pin) * 1e3, mg = median(ts_page) * 1e3;
        double mind = *std::min_element(ts_dev.begin(), ts_dev.end()) * 1e3;
        std::string zs; for (int iz : sys) zs += std::to_string(zones[iz].id) + " ";
        printf("SYS zones=[%s] n=%d nnz=%d info=%d npiv=%d lu_nnz=%ld | analysis %.1f ms, factor %.1f ms (again %.1f ms) | "
               "solve dev %.3f ms (min %.3f), +copies pinned %.3f ms, pageable %.3f ms | mem: est dev %.1f MB (peak %.1f), "
               "measured %.1f MB | worst rel %.1e res %.1e (mkl %.1e)\n",
               zs.c_str(), n, nnz, info, npiv, (long)lunnz, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t4 - t3) * 1e3, md, mind,
               mp, mg, mem[0] / 1e6, mem[1] / 1e6, ((double)free0 - (double)free1) / 1e6, worst_rel, worst_res, worst_res_mkl);
        fflush(stdout);
        sum_solve += md; sum_solve_pin += mp; sum_solve_page += mg; sum_an += t1 - t0; sum_fa += t2 - t1;

        cudaFreeHost(hb_pin); cudaFreeHost(hx_pin);
        cudssMatrixDestroy(A); cudssMatrixDestroy(B); cudssMatrixDestroy(X);
        cudssDataDestroy(handle, data); cudssConfigDestroy(cfg);
        cudaFree(d_ia); cudaFree(d_ja); cudaFree(d_a); cudaFree(d_b); cudaFree(d_x);
    }
    printf("TOTAL mode=%s layout=%s: analysis %.1f ms, factor %.1f ms, one pressure solve (all zones): dev %.3f ms, "
           "pinned %.3f ms, pageable %.3f ms\n",
           mode.c_str(), layout.c_str(), sum_an * 1e3, sum_fa * 1e3, sum_solve, sum_solve_pin, sum_solve_page);
    cudssDestroy(handle);
    return 0;
}
