import struct


def indicator(x, y, z):
    return (x - x0)**2 / rx**2 + (y - y0)**2 / ry**2 + (z - z0)**2 / rz**2 - 1


shift = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1),
         (1, 1, 0), (1, 1, 1))

x0 = 1 / 2
y0 = 1 / 2
z0 = 1 / 2
rx = 1 / 4
ry = 1 / 5
rz = 1 / 6
minlevel = 1
maxlevel = 6
L = 1 << 32
C = []


def refinep(x, y, z, delta):
    seen = None
    for dx, dy, dz in shift:
        u = (x + delta * dx) / L
        v = (y + delta * dy) / L
        w = (z + delta * dz) / L
        sign = indicator(u, v, w) > 0
        if sign < 0:
            print(sign)
        if seen != None and sign != seen:
            return True
        else:
            seen = sign
    return False


def traverse(x, y, z, level):
    delta = 1 << (32 - level)
    if level + 1 <= minlevel or (level + 1 <= maxlevel
                                 and refinep(x, y, z, delta)):
        d = 1 << (32 - level - 1)
        for dx, dy, dz in shift:
            traverse(x + d * dx, y + d * dy, z + d * dz, level + 1)
    else:
        C.append((x, y, z, level))


traverse(0, 0, 0, 0)
m_level = max(C[3] for C in C)
s_level = 32 - m_level
size = 1 << m_level
with open("in.cells",
          "wb") as cell, open("in.scalars",
                              "wb") as scalars, open("in.field",
                                                     "wb") as field:
    for x, y, z, level in C:
        cell.write(struct.pack("iii", x >> s_level, y >> s_level,
                               z >> s_level))
        cell.write(struct.pack("i", m_level - level))
        delta = 1 << (32 - level - 1)
        u = (x + delta) / L
        v = (y + delta) / L
        w = (z + delta) / L
        scalars.write(struct.pack("f", indicator(u, v, w)))
        field.write(struct.pack("f", u))
