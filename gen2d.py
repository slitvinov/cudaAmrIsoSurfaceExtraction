import struct
import sys

nx = 20
ny = 20
level = 0
x0 = nx / 2
y0 = ny / 2
rx = nx / 4
ry = nx / 5


def indicator(x, y):
    return (x - x0)**2 / rx**2 + (y - y0)**2 / ry**2 - 1


with open("in.cells",
          "wb") as cell, open("in.scalar",
                              "wb") as scalars, open("in.field",
                                                     "wb") as field:
    for x in range(nx):
        for y in range(ny):
            cell.write(struct.pack("ii", x, y))
            cell.write(struct.pack("i", level))
            scalars.write(struct.pack("f", indicator(x, y)))
            field.write(struct.pack("f", x))
sys.stderr.write("gen2d.py: write in.cells in.scalar in.field\n")
