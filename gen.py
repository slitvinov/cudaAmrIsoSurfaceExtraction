import struct

nx = 10
ny = 10
nz = 10
level = 0
x0 = nx / 2
y0 = ny / 2
z0 = nz / 2
rx = nx / 4
ry = nx / 5
rz = nx / 6
with open("in.cells", "wb") as cell, open("in.scalars", "wb") as scalars, open("in.field", "wb") as field:
    for x in range(0, nx):
        for y in range(0, ny):
            for z in range(0, nz):
                cell.write(struct.pack("iii", x, y, z))
                cell.write(struct.pack("i", level))
                r = (x - x0)**2 / rx**2 + (y - y0)**2 / ry**2 + (z -
                                                                 z0)**2 / rz**2
                scalars.write(struct.pack("f", r - 1))
                field.write(struct.pack("f", x))
