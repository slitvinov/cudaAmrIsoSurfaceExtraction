import struct
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 3:
    sys.stderr.write("usage: view2d.py prefix in.cells\n")
    sys.exit(1)
prefix = sys.argv[1]
cells_path = sys.argv[2]

with open(prefix + ".xy.raw", "rb") as f:
    data = f.read()
nvert = len(data) // (2 * 4)
xy = np.array(struct.unpack("<%df" % (2 * nvert), data)).reshape(nvert, 2)

with open(prefix + ".seg.raw", "rb") as f:
    data = f.read()
nseg = len(data) // (2 * 4)
seg = np.array(struct.unpack("<%di" % (2 * nseg), data)).reshape(nseg, 2)

with open(prefix + ".attr.raw", "rb") as f:
    data = f.read()
attr = np.array(struct.unpack("<%df" % nvert, data))

with open(cells_path, "rb") as f:
    cdata = f.read()
nc = len(cdata) // (3 * 4)
cells = np.array(struct.unpack("<%di" % (3 * nc), cdata)).reshape(nc, 3)

fig, ax = plt.subplots(figsize=(8, 8))
for c in cells:
    x, y, lev = c
    h = 1 << lev
    rect = plt.Rectangle((x, y), h, h, linewidth=0.3, edgecolor="0.7", facecolor="none")
    ax.add_patch(rect)
for s in seg:
    ax.plot(xy[s, 0], xy[s, 1], "k-", linewidth=2)
sc = ax.scatter(xy[:, 0], xy[:, 1], c=attr, s=10, zorder=5)
plt.colorbar(sc, label="u")
ax.set_aspect("equal")
ax.set_title("nseg=%d nvert=%d" % (nseg, nvert))
plt.savefig(prefix + ".svg", bbox_inches="tight")
sys.stderr.write("view2d.py: %s.svg\n" % prefix)
