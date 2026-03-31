import struct
import sys
import collections


def indicator(x, y):
    return (x - x0)**2 / rx**2 + (y - y0)**2 / ry**2 - 1


shift = ((0, 0), (0, 1), (1, 0), (1, 1))

x0 = 1 / 2
y0 = 1 / 2
rx = 1 / 4
ry = 1 / 5
minlevel = 1
maxlevel = 6
L = 1 << 32
C = []


def refinep(x, y, delta):
    seen = None
    for dx, dy in shift:
        u = (x + delta * dx) / L
        v = (y + delta * dy) / L
        sign = indicator(u, v) > 0
        if seen != None and sign != seen:
            return True
        else:
            seen = sign
    return False


def traverse(x, y, level):
    delta = 1 << (32 - level)
    if level + 1 <= minlevel or (level + 1 <= maxlevel
                                 and refinep(x, y, delta)):
        d = 1 << (32 - level - 1)
        for dx, dy in shift:
            traverse(x + d * dx, y + d * dy, level + 1)
    else:
        C.append((x, y, level))


traverse(0, 0, 0)
m_level = max(c[2] for c in C)
s_level = 32 - m_level
size = 1 << m_level
stat = collections.Counter()
with open("in.cells",
          "wb") as cell, open("in.scalar",
                              "wb") as scalars, open("in.field",
                                                     "wb") as field:
    for x, y, level in C:
        cell.write(struct.pack("ii", x >> s_level, y >> s_level))
        cell.write(struct.pack("i", m_level - level))
        stat[m_level - level] += 1
        delta = 1 << (32 - level - 1)
        u = (x + delta) / L
        v = (y + delta) / L
        scalars.write(struct.pack("f", indicator(u, v)))
        field.write(struct.pack("f", u))
sys.stderr.write("gen2d-amr.py: write in.cells in.scalar in.field\n")
print(stat)
