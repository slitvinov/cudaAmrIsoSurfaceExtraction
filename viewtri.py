import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if len(sys.argv) < 2:
    sys.stderr.write("usage: viewtri.py prefix\n")
    sys.exit(1)
prefix = sys.argv[1]

with open(prefix + ".xyz.raw", "rb") as f:
    data = f.read()
nvert = len(data) // (3 * 4)
xyz = np.array(struct.unpack("<%df" % (3 * nvert), data)).reshape(nvert, 3)

with open(prefix + ".tri.raw", "rb") as f:
    data = f.read()
ntri = len(data) // (3 * 4)
tri = np.array(struct.unpack("<%di" % (3 * ntri), data)).reshape(ntri, 3)

with open(prefix + ".attr.raw", "rb") as f:
    data = f.read()
attr = np.array(struct.unpack("<%df" % nvert, data))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
tri_verts = xyz[tri]
tri_attr = attr[tri].mean(axis=1)
norm = plt.Normalize(tri_attr.min(), tri_attr.max())
colors = plt.cm.viridis(norm(tri_attr))
poly = Poly3DCollection(tri_verts,
                        facecolors=colors,
                        edgecolor="k",
                        linewidth=0.1)
ax.add_collection3d(poly)
ax.auto_scale_xyz(xyz[:, 0], xyz[:, 1], xyz[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
plt.colorbar(sm, ax=ax, shrink=0.6, label="u")
ax.set_title("ntri=%d nvert=%d" % (ntri, nvert))
plt.savefig(prefix + ".png", dpi=150, bbox_inches="tight")
sys.stderr.write("viewtri.py: %s.png\n" % prefix)
