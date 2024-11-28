import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mesh = meshio.read("mesh.xdmf2", "xdmf")
points = mesh.points
cells = mesh.cells_dict.get("triangle")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
triangles = [points[cell] for cell in cells]
poly_collection = Poly3DCollection(triangles, edgecolor="k")
ax.add_collection3d(poly_collection)
plt.savefig("mesh.png")
