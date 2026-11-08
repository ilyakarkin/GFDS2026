#!/usr/bin/env python3
"""Generate radinit_many_meshes.fds: a synthetic building for timing the initial radiation solves.

A block of storeys with rooms (one cell thick walls with door openings), furniture boxes, a few hot
panels and a hot gas layer, split over many meshes, with an OPEN exterior.  Radiation is on by default
(finite volume method, 104 angles), so the set-up runs INITIAL_RADIATION_ITERATIONS=3 full solves.
Run it with a tiny T_END: only the set-up and the first time step matter.  VERBOSE prints the wall
clock time of 'Completed radiation initialization'.

    python3 radinit_many_meshes.py [NX NY NZ [CELLS_PER_MESH_XY CELLS_PER_MESH_Z [BOXES_PER_ROOM]]] > radinit_many_meshes.fds

Defaults 4 4 2 48 42 9: 32 meshes, 3.1 million cells, about 3700 OBSTs.
"""
import sys

def main():
    a = [int(x) for x in sys.argv[1:]]
    nmx, nmy, nmz = (a[0:3] if len(a) >= 3 else (4, 4, 2))   # meshes in x, y, z
    cxy, cz = (a[3:5] if len(a) >= 5 else (48, 42))         # cells per mesh in x/y and z
    nbox = a[5] if len(a) >= 6 else 9                        # furniture boxes per room (at most 9)
    dx = 0.1
    lx, ly, lz = nmx*cxy*dx, nmy*cxy*dx, nmz*cz*dx
    room, storey, margin = 2.0, 2.8, 0.6                     # room pitch, storey height, air around the block (m)
    out = []
    w = out.append
    w("&HEAD CHID='radinit_many_meshes', TITLE='Many meshes and OBSTs with radiation (initial radiation timing)' /")
    w("")
    w("&TIME T_END=0.0001 /")
    w("&MISC VERBOSE=.TRUE. /")
    w("&DUMP NFRAMES=1, SMOKE3D=.FALSE. /")
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
    w("&SURF ID='HOT', TMP_FRONT=600., COLOR='RED' /")
    w("&MATL ID='GYPSUM', CONDUCTIVITY=0.48, SPECIFIC_HEAT=0.84, DENSITY=1440. /")
    w("&SURF ID='WALL', MATL_ID='GYPSUM', THICKNESS=0.1, COLOR='GRAY' /")
    w("")
    x0, x1 = margin, lx - margin
    y0, y1 = margin, ly - margin
    nrx, nry = int((x1-x0)/room + 1e-6), int((y1-y0)/room + 1e-6)
    nst = max(1, int((lz - 0.3)/storey + 1e-6))
    x1, y1 = x0 + nrx*room, y0 + nry*room
    t = dx                                                 # wall and slab thickness: one cell
    door = 0.8                                             # door width; doors are 2 m high
    n_obst = 0
    def obst(xb, surf="WALL"):
        nonlocal n_obst
        n_obst += 1
        w("&OBST XB=%.2f,%.2f,%.2f,%.2f,%.2f,%.2f, SURF_ID='%s' /" % (tuple(xb) + (surf,)))
    for s in range(nst + 1):                               # slabs, ground floor included; a stair hole in each
        z = s*storey
        if s == 0:
            obst((x0, x1+t, y0, y1+t, z, z+t))
        else:
            obst((x0, x0+room, y0, y1+t, z, z+t))
            obst((x0+room, x1+t, y0, y1+t, z, z+t))
    for s in range(nst):
        za, zb = s*storey + t, (s+1)*storey
        zd = min(zb, za + 2.0)
        for i in range(nrx + 1):                           # walls normal to x, with a door in each room
            x = x0 + i*room
            for j in range(nry):
                ya = y0 + j*room
                yd = ya + 0.6
                obst((x, x+t, ya, yd, za, zb))
                obst((x, x+t, yd+door, ya+room+t, za, zb))
                obst((x, x+t, yd, yd+door, zd, zb))
        for j in range(nry + 1):                           # walls normal to y, with a door in each room
            y = y0 + j*room
            for i in range(nrx):
                xa = x0 + i*room
                xd = xa + 0.6
                obst((xa, xd, y, y+t, za, zb))
                obst((xd+door, xa+room+t, y, y+t, za, zb))
                obst((xd, xd+door, y, y+t, zd, zb))
        for j in range(nry):
            for i in range(nrx):
                xa, ya = x0 + i*room, y0 + j*room
                for b in range(min(nbox, 9)):              # furniture: boxes on a 3 x 3 grid
                    bx, by = xa + (0.2, 0.8, 1.4)[b % 3], ya + (0.2, 0.8, 1.4)[b // 3]
                    hot = (i + 2*j + 3*s + b) % 23 == 0
                    obst((bx, bx+0.3, by, by+0.3, za, za+0.4 + 0.2*(b % 4)), "HOT" if hot else "WALL")
    w("")
    w("&INIT XB=%.2f,%.2f,%.2f,%.2f,%.2f,%.2f, TEMPERATURE=400. /" % (x0, x0+4*room, y0, y0+3*room, storey-0.8, storey))
    w("")
    for s in range(nst):                                   # radiometers on the floor and in the gas
        za = s*storey + t
        for j in range(0, nry, 2):
            for i in range(0, nrx, 2):
                xc, yc = x0 + i*room + 0.67, y0 + j*room + 0.67  # floor between the furniture boxes
                w("&DEVC ID='RHF_%d_%d_%d', XYZ=%.2f,%.2f,%.2f, QUANTITY='RADIATIVE HEAT FLUX', IOR=3 /" % (s, i, j, xc, yc, za))
                w("&DEVC ID='U_%d_%d_%d', XYZ=%.2f,%.2f,%.2f, QUANTITY='INTEGRATED INTENSITY' /" % (s, i, j, xc, yc, za+1.27))
    w("")
    w("&SLCF PBZ=%.2f, QUANTITY='INTEGRATED INTENSITY' /" % (storey - 0.45))
    w("&BNDF QUANTITY='RADIATIVE HEAT FLUX' /")
    w("")
    w("&TAIL /")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stderr.write("%d meshes, %d cells, %d storeys, %d x %d rooms, %d OBSTs\n" %
                     (nmx*nmy*nmz, nmx*nmy*nmz*cxy*cxy*cz, nst, nrx, nry, n_obst))

if __name__ == "__main__":
    main()
