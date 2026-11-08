#!/usr/bin/env python3
"""Generate pz_many_meshes.fds: a synthetic building for timing INITIALIZE_PRESSURE_ZONES.

A block of sealed rooms (one cell thick walls and slabs) split over many meshes, with
thin (zero thickness) partitions in some rooms, furniture boxes, and an OPEN exterior.
None of the rooms has a ZONE line: all pressure zones are found automatically, as in
user models.  Run it with a tiny T_END: only the set-up matters.

    python3 pz_many_meshes.py [NX NY NZ [CELLS_PER_MESH_XY CELLS_PER_MESH_Z [BOXES_PER_ROOM]]] > pz_many_meshes.fds

Defaults 8 4 2 40 30 6: 64 meshes, 3.1 million cells, 210 rooms, 315 pressure zones, 1832 OBSTs.
"""
import sys

def main():
    a = [int(x) for x in sys.argv[1:]]
    nmx, nmy, nmz = (a[0:3] if len(a) >= 3 else (8, 4, 2))   # meshes in x, y, z
    cxy, cz = (a[3:5] if len(a) >= 5 else (40, 30))         # cells per mesh in x/y and z
    nbox = a[5] if len(a) >= 6 else 6                        # furniture boxes per room (at most 9)
    dx = 0.1
    lx, ly, lz = nmx*cxy*dx, nmy*cxy*dx, nmz*cz*dx
    room, storey, margin = 2.0, 2.8, 1.0                     # room pitch, storey height, air around the block (m)
    out = []
    w = out.append
    w("&HEAD CHID='pz_many_meshes', TITLE='Many meshes and automatically found pressure zones (set-up timing)' /")
    w("")
    w("&TIME T_END=0.0001 /")
    w("&DUMP NFRAMES=1, SMOKE3D=.FALSE. /")
    w("&RADI RADIATION=.FALSE. /")
    w("")
    for k in range(nmz):
        for j in range(nmy):
            for i in range(nmx):
                w("&MESH IJK=%d,%d,%d, XB=%.2f,%.2f,%.2f,%.2f,%.2f,%.2f /" %
                  (cxy, cxy, cz, i*cxy*dx, (i+1)*cxy*dx, j*cxy*dx, (j+1)*cxy*dx, k*cz*dx, (k+1)*cz*dx))
    w("")
    w("&VENT DB='XMIN', SURF_ID='OPEN' /")
    w("&VENT DB='XMAX', SURF_ID='OPEN' /")
    w("&VENT DB='YMIN', SURF_ID='OPEN' /")
    w("&VENT DB='YMAX', SURF_ID='OPEN' /")
    w("&VENT DB='ZMAX', SURF_ID='OPEN' /")
    w("")
    x0, x1 = margin, lx - margin
    y0, y1 = margin, ly - margin
    nrx, nry = int((x1-x0)/room + 1e-6), int((y1-y0)/room + 1e-6)
    nst = max(1, int((lz - 0.3)/storey + 1e-6))
    x1, y1 = x0 + nrx*room, y0 + nry*room
    zt = nst*storey
    t = dx                                                 # wall and slab thickness: one cell
    n_obst = 0
    def obst(xb, extra=""):
        nonlocal n_obst
        n_obst += 1
        w("&OBST XB=%.2f,%.2f,%.2f,%.2f,%.2f,%.2f, SURF_ID='INERT'%s /" % (tuple(xb) + (extra,)))
    for s in range(nst + 1):                               # slabs, ground floor included
        z = s*storey
        obst((x0, x1+t, y0, y1+t, z, z+t))
    for s in range(nst):
        za, zb = s*storey, (s+1)*storey
        for i in range(nrx + 1):                           # walls normal to x, one per room row so they stay separate OBSTs
            x = x0 + i*room
            for j in range(nry):
                obst((x, x+t, y0+j*room, y0+(j+1)*room+t, za, zb))
        for j in range(nry + 1):                           # walls normal to y
            y = y0 + j*room
            for i in range(nrx):
                obst((x0+i*room, x0+(i+1)*room+t, y, y+t, za, zb))
        for j in range(nry):
            for i in range(nrx):
                xa, ya = x0 + i*room, y0 + j*room
                c = (i + 3*j + 7*s) % 4
                if c == 0:                                 # thin partition (zero thickness): splits the room in two zones
                    obst((xa+1.0, xa+1.0, ya+t, ya+room, za+t, zb))
                elif c == 1:                               # thin partition across y
                    obst((xa+t, xa+room, ya+1.0, ya+1.0, za+t, zb))
                for b in range(min(nbox, 9)):              # furniture: boxes on a 3 x 3 grid, clear of the partitions
                    bx, by = xa + (0.2, 0.6, 1.3)[b % 3], ya + (0.2, 0.6, 1.3)[b // 3]
                    obst((bx, bx+0.3, by, by+0.3, za+t, za+0.4 + 0.2*(b % 4)))
    w("")
    w("&TAIL /")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stderr.write("meshes %d, cells %d, rooms %d x %d x %d, OBST %d\n" %
                     (nmx*nmy*nmz, nmx*nmy*nmz*cxy*cxy*cz, nrx, nry, nst, n_obst))

main()
