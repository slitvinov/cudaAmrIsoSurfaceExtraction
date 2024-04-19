#include "table.inc"
#include <cuda.h>
#include <stdio.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

struct vec3i {
  int x, y, z;
};

__device__ vec3i operator>>(const vec3i v, const int s) {
  vec3i u;
  u.x = v.x >> s;
  u.y = v.y >> s;
  u.z = v.z >> s;
  return u;
}

__device__ __host__ long leftShift3(long x) {
  x = (x | x << 32) & 0x1f00000000ffffull;
  x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full;
  x = (x | x << 4) & 0x10c30c30c30c30c3ull;
  x = (x | x << 2) & 0x1249249249249249ull;
  return x;
}

__device__ __host__ long mortonCode(int x, int y, int z) {
  return (leftShift3(uint32_t(z)) << 2) | (leftShift3(uint32_t(y)) << 1) |
         (leftShift3(uint32_t(x)) << 0);
}

struct vec3f {
  __device__ vec3f() {}
  __device__ vec3f(const float x, const float y, const float z)
      : x(x), y(y), z(z) {}
  __device__ vec3f(const vec3i o) : x(o.x), y(o.y), z(o.z) {}
  float x, y, z;
};

struct TriangleVertex {
  vec3f position;
  float scalar, field;
  uint32_t id;
};

__device__ bool operator==(const vec3f &a, const vec3f &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ bool operator<(const vec3f &a, const vec3f &b) {
  return (a.x < b.x) ||
         ((a.x == b.x) && ((a.y < b.y) || (a.y == b.y) && (a.z < b.z)));
}

struct Cell {
  vec3i lower;
  int level;
  uint64_t morton;
  float scalar, field;
};

__device__ struct TriangleVertex dual(struct Cell c) {
  struct TriangleVertex v;
  v.position.x = c.lower.x + 0.5 * (1 << c.level);
  v.position.y = c.lower.y + 0.5 * (1 << c.level);
  v.position.z = c.lower.z + 0.5 * (1 << c.level);
  v.scalar = c.scalar;
  v.field = c.field;
  return v;
}

struct CompareMorton0 {
  __device__ bool operator()(const Cell &a, const uint64_t b) {
    return a.morton < b;
  }
};

struct CompareVertices {
  __device__ bool operator()(const TriangleVertex &a,
                             const TriangleVertex &b) const {
    return a.position < b.position;
  }
};

struct AMR {
  __device__ AMR(const Cell *const __restrict__ cellArray, const int ncell)
      : cellArray(cellArray), ncell(ncell) {}

  __device__ bool findActual(struct Cell &result, const vec3i lower,
                             int level) {
    const Cell *const __restrict__ begin = cellArray;
    const Cell *const __restrict__ end = cellArray + ncell;
    const Cell *it = thrust::system::detail::generic::scalar::lower_bound(
        cellArray, cellArray + ncell, mortonCode(lower.x, lower.y, lower.z),
        CompareMorton0());
    if (it == end)
      return false;
    const Cell found = *it;
    if ((found.lower >> max(level, found.level)) ==
        (lower >> max(level, found.level))) {
      result = found;
      return true;
    }

    if (it > begin) {
      const Cell found = it[-1];
      if ((found.lower >> max(level, found.level)) ==
          (lower >> max(level, found.level))) {
        result = found;
        return true;
      }
    }
    return false;
  };

  const Cell *const __restrict__ cellArray;
  const int ncell;
};

__global__ void extractTriangles(const Cell *const __restrict__ cellArray,
                                 int ncell, int maxlevel, float iso,
                                 TriangleVertex *__restrict__ out, int size,
                                 int *cnt) {
  size_t tid;
  int x, y, z, id, index, i, j, k, ii, wid, did, dx, dy, dz, ix, iy, iz;
  int8_t *edge, *vert;
  float t;
  TriangleVertex v0, v1, triVertex[3];
  vec3i lower;
  Cell corner[2][2][2], cell;
  AMR amr(cellArray, ncell);
  tid = threadIdx.x + size_t(blockDim.x) * blockIdx.x;
  wid = tid / 8;
  if (wid >= ncell)
    return;
  did = tid % 8;
  cell = cellArray[wid];
  dz = (did & 4) ? 1 : -1;
  dy = (did & 2) ? 1 : -1;
  dx = (did & 1) ? 1 : -1;
  for (iz = 0; iz < 2; iz++)
    for (iy = 0; iy < 2; iy++)
      for (ix = 0; ix < 2; ix++) {
        lower.x = cell.lower.x + dx * ix * (1 << cell.level);
        lower.y = cell.lower.y + dy * iy * (1 << cell.level);
        lower.z = cell.lower.z + dz * iz * (1 << cell.level);
        if (!amr.findActual(corner[iz][iy][ix], lower, cell.level))
          return;
        if (corner[iz][iy][ix].level < cell.level)
          return;
        if (corner[iz][iy][ix].level == cell.level &&
            corner[iz][iy][ix].lower < cell.lower)
          return;
      }
  x = dx == -1;
  y = dy == -1;
  z = dz == -1;
  TriangleVertex vertex[8] = {
      dual(corner[0 + z][0 + y][0 + x]), dual(corner[0 + z][0 + y][1 - x]),
      dual(corner[0 + z][1 - y][1 - x]), dual(corner[0 + z][1 - y][0 + x]),
      dual(corner[1 - z][0 + y][0 + x]), dual(corner[1 - z][0 + y][1 - x]),
      dual(corner[1 - z][1 - y][1 - x]), dual(corner[1 - z][1 - y][0 + x])};
  index = 0;
  for (i = 0; i < 8; i++)
    if (vertex[i].scalar > iso)
      index += (1 << i);
  if (index == 0 || index == 0xff)
    return;
  for (edge = &vtkMarchingCubesTriangleCases[index][0]; edge[0] > -1;
       edge += 3) {
    for (ii = 0; ii < 3; ii++) {
      vert = vtkMarchingCubes_edges[edge[ii]];
      v0 = vertex[vert[0]];
      v1 = vertex[vert[1]];
      t = (iso - v0.scalar) / float(v1.scalar - v0.scalar);

      triVertex[ii].position.x = (1.0 - t) * v0.position.x + t * v1.position.x;
      triVertex[ii].position.y = (1.0 - t) * v0.position.y + t * v1.position.y;
      triVertex[ii].position.z = (1.0 - t) * v0.position.z + t * v1.position.z;
      triVertex[ii].scalar = (1.0 - t) * v0.scalar + t * v1.scalar;
      triVertex[ii].field = (1.0 - t) * v0.field + t * v1.field;
    }
    if (triVertex[1].position == triVertex[0].position)
      continue;
    if (triVertex[2].position == triVertex[0].position)
      continue;
    if (triVertex[1].position == triVertex[2].position)
      continue;
    id = atomicAdd(cnt, 1);
    if (id >= 3 * size)
      continue;
    for (j = 0; j < 3; j++) {
      k = 3 * id + j;
      out[k].position.x = triVertex[j].position.x;
      out[k].position.y = triVertex[j].position.y;
      out[k].position.z = triVertex[j].position.z;
      out[k].scalar = triVertex[j].scalar;
      out[k].field = triVertex[j].field;
      out[k].id = 4 * id + j;
    }
  }
}

__global__ void
createVertexArray(int *cnt, const TriangleVertex *const __restrict__ vertices,
                  int nvert, TriangleVertex *vert, int size, int3 *index) {
  int i, j, k, l, id, tid, *tri;
  TriangleVertex vertex;
  tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nvert)
    return;
  vertex = vertices[tid];
  if (tid > 0 && vertex.position == vertices[tid - 1].position)
    return;
  id = atomicAdd(cnt, 1);
  if (id >= size)
    return;
  vert[id].position.x = vertex.position.x;
  vert[id].position.y = vertex.position.y;
  vert[id].position.z = vertex.position.z;
  vert[id].scalar = vertex.scalar;
  vert[id].field = vertex.field;

  for (i = tid; i < nvert && vertices[i].position == vertex.position; i++) {
    j = vertices[i].id;
    k = j % 4;
    l = j / 4;
    tri = &index[l].x;
    tri[k] = id;
  }
}

static int comp(const void *av, const void *bv) {
  struct Cell *a, *b;
  a = (struct Cell *)av;
  b = (struct Cell *)bv;
  return a->morton > b->morton;
}

int main(int argc, char **argv) {
  float iso, *attr, xyz[3];
  TriangleVertex *vert;
  int3 *tri;
  size_t numJobs;
  int Verbose, maxlevel, blockSize, numBlocks;
  long i, j, nvert, ntri, ncell, size;
  FILE *file, *cell_file, *scalar_file, *field_file;
  int cell[4], ox, oy, oz;
  char attr_path[FILENAME_MAX], xyz_path[FILENAME_MAX], tri_path[FILENAME_MAX],
      xdmf_path[FILENAME_MAX], *attr_base, *xyz_base, *tri_base, *cell_path,
      *scalar_path, *field_path, *output_path, *end;
  struct Cell *cells;

  Verbose = 0;
  while (*++argv != NULL && argv[0][0] == '-')
    switch (argv[0][1]) {
    case 'h':
      fprintf(stderr, "Usage: iso [-v] in.cells in.scalar in.field iso mesh\n");
      exit(1);
    case 'v':
      Verbose = 1;
      break;
    case '-':
      argv++;
      goto positional;
    default:
      fprintf(stderr, "iso: error: unknown option '%s'\n", *argv);
      exit(1);
    }
positional:
  if ((cell_path = *argv++) == NULL) {
    fprintf(stderr, "iso: error: in.cells is not given\n");
    exit(1);
  }
  if ((scalar_path = *argv++) == NULL) {
    fprintf(stderr, "iso: error: in.scalar is not given\n");
    exit(1);
  }
  if ((field_path = *argv++) == NULL) {
    fprintf(stderr, "iso: error: in.field is not given\n");
    exit(1);
  }
  if (*argv == NULL) {
    fprintf(stderr, "iso: error: iso is no given\n");
    exit(1);
  }
  iso = strtod(*argv, &end);
  if (*end != '\0') {
    fprintf(stderr, "iso: error: '%s' is not a number\n", *argv);
    exit(1);
  }
  argv++;
  if ((output_path = *argv++) == NULL) {
    fprintf(stderr, "iso: error: out.mesh is not given\n");
    exit(1);
  }
  if ((cell_file = fopen(cell_path, "r")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", cell_path);
    exit(1);
  }
  if ((scalar_file = fopen(scalar_path, "r")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", scalar_path);
    exit(1);
  }
  if ((field_file = fopen(field_path, "r")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", field_path);
    exit(1);
  }
  fseek(cell_file, 0, SEEK_END);
  size = ftell(cell_file);
  fseek(cell_file, 0, SEEK_SET);
  ncell = size / (4 * sizeof(int));
  if ((cells = (struct Cell *)malloc(ncell * sizeof *cells)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  ox = INT_MAX;
  oy = INT_MAX;
  oz = INT_MAX;
  maxlevel = 0;
  for (i = 0; i < ncell; i++) {
    if (fread(cell, sizeof(cell), 1, cell_file) != 1) {
      fprintf(stderr, "iso: error: fail to read '%s'\n", cell_path);
      exit(1);
    }
    cells[i].lower.x = cell[0];
    cells[i].lower.y = cell[1];
    cells[i].lower.z = cell[2];
    cells[i].level = cell[3];
    if (fread(&cells[i].scalar, sizeof(cells[i].scalar), 1, scalar_file) != 1) {
      fprintf(stderr, "iso: error: fail to read '%s'\n", scalar_path);
      exit(1);
    }
    if (fread(&cells[i].field, sizeof(cells[i].field), 1, field_file) != 1) {
      fprintf(stderr, "iso: error: fail to read '%s'\n", field_path);
      exit(1);
    }
    maxlevel = std::max(maxlevel, cells[i].level);
    ox = std::min(ox, cells[i].lower.x);
    oy = std::min(oy, cells[i].lower.y);
    oz = std::min(oz, cells[i].lower.z);
  }
  for (i = 0; i < ncell; i++) {
    cells[i].lower.x -= ox;
    cells[i].lower.y -= oy;
    cells[i].lower.z -= oz;
    cells[i].morton =
        mortonCode(cells[i].lower.x, cells[i].lower.y, cells[i].lower.z);
  }

  if (Verbose)
    fprintf(stderr, "iso: ncell, maxlevel, origin: %ld %d [%d %d %d]\n", ncell,
            maxlevel, ox, oy, oz);
  if (fclose(cell_file) != 0) {
    fprintf(stderr, "iso: error: fail to close '%s'\n", cell_path);
    exit(1);
  }
  if (fclose(scalar_file) != 0) {
    fprintf(stderr, "iso: error: fail to close '%s'\n", scalar_path);
    exit(1);
  }
  if (fclose(field_file) != 0) {
    fprintf(stderr, "iso: error: fail to close '%s'\n", field_path);
    exit(1);
  }
  qsort(cells, ncell, sizeof *cells, comp);
  struct Cell *d_cells;
  cudaMalloc(&d_cells, ncell * sizeof *d_cells);
  cudaMemcpy(d_cells, cells, ncell * sizeof *d_cells, cudaMemcpyHostToDevice);
  thrust::device_vector<int> d_atomicCounter(1);
  thrust::device_vector<TriangleVertex> d_triangleVertices(0);
  d_atomicCounter[0] = 0;
  numJobs = 8 * ncell;
  blockSize = 512;
  numBlocks = (numJobs + blockSize - 1) / blockSize;
  extractTriangles<<<numBlocks, blockSize>>>(
      d_cells, ncell, maxlevel, iso, NULL, 0,
      thrust::raw_pointer_cast(d_atomicCounter.data()));
  cudaDeviceSynchronize();
  ntri = d_atomicCounter[0];
  d_triangleVertices.resize(3 * ntri);
  d_atomicCounter[0] = 0;
  extractTriangles<<<numBlocks, blockSize>>>(
      d_cells, ncell, maxlevel, iso,
      thrust::raw_pointer_cast(d_triangleVertices.data()), 3 * ntri,
      thrust::raw_pointer_cast(d_atomicCounter.data()));
  cudaDeviceSynchronize();
  cudaFree(d_cells);
  try {
    thrust::sort(d_triangleVertices.begin(), d_triangleVertices.end(),
                 CompareVertices());
    cudaDeviceSynchronize();
  } catch (thrust::system::system_error) {
    fprintf(stderr, "iso: thrust::sort failed\n");
    exit(1);
  }
  thrust::device_vector<TriangleVertex> d_vert(0);
  thrust::device_vector<int3> d_tri(ntri);
  d_atomicCounter[0] = 0;
  numJobs = 3 * ntri;
  numBlocks = (numJobs + blockSize - 1) / blockSize;
  createVertexArray<<<numBlocks, blockSize>>>(
      thrust::raw_pointer_cast(d_atomicCounter.data()),
      thrust::raw_pointer_cast(d_triangleVertices.data()), 3 * ntri, NULL, 0,
      thrust::raw_pointer_cast(d_tri.data()));
  cudaDeviceSynchronize();
  nvert = d_atomicCounter[0];
  d_vert.resize(nvert);
  d_atomicCounter[0] = 0;
  createVertexArray<<<numBlocks, blockSize>>>(
      thrust::raw_pointer_cast(d_atomicCounter.data()),
      thrust::raw_pointer_cast(d_triangleVertices.data()), 3 * ntri,
      thrust::raw_pointer_cast(d_vert.data()), nvert,
      thrust::raw_pointer_cast(d_tri.data()));
  cudaDeviceSynchronize();
  if (Verbose)
    fprintf(stderr, "iso: nvert: %ld\n", nvert);
  assert(d_tri.size() == ntri);
  assert(d_vert.size() == nvert);
  if ((vert = (TriangleVertex *)malloc(nvert * sizeof *vert)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  if ((tri = (int3 *)malloc(ntri * sizeof *tri)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  thrust::copy(d_vert.begin(), d_vert.end(), vert);
  thrust::copy(d_tri.begin(), d_tri.end(), tri);

  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", output_path);
  snprintf(tri_path, sizeof tri_path, "%s.tri.raw", output_path);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", output_path);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", output_path);
  xyz_base = xyz_path;
  tri_base = tri_path;
  attr_base = attr_path;
  for (j = 0; xyz_path[j] != '\0'; j++) {
    if (xyz_path[j] == '/' && xyz_path[j + 1] != '\0') {
      xyz_base = &xyz_path[j + 1];
      tri_base = &tri_path[j + 1];
      attr_base = &attr_path[j + 1];
    }
  }
  if ((file = fopen(xyz_path, "w")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", xyz_path);
    exit(1);
  }
  for (i = 0; i < nvert; i++) {
    xyz[0] = vert[i].position.x;
    xyz[1] = vert[i].position.y;
    xyz[2] = vert[i].position.z;
    if (fwrite(xyz, sizeof xyz, 1, file) != 1) {
      fprintf(stderr, "iso: error: fail to write '%s'\n", xyz_path);
      exit(1);
    }
  }
  if (fclose(file) != 0) {
    fprintf(stderr, "iso: fail to close '%s'\n", xyz_path);
    exit(1);
  }
  if ((file = fopen(tri_path, "w")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", tri_path);
    exit(1);
  }
  if (fwrite(tri, ntri * sizeof *tri, 1, file) != 1) {
    fprintf(stderr, "iso: error: fail to write '%s'\n", tri_path);
    exit(1);
  }
  if (fclose(file) != 0) {
    fprintf(stderr, "iso: fail to close '%s'\n", tri_path);
    exit(1);
  }
  if ((attr = (float *)malloc(nvert * sizeof *attr)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  for (i = 0; i < nvert; i++)
    attr[i] = vert[i].field;
  if ((file = fopen(attr_path, "w")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", attr_path);
    exit(1);
  }
  if (fwrite(attr, nvert * sizeof *attr, 1, file) != 1) {
    fprintf(stderr, "iso: error: fail to write '%s'\n", attr_path);
    exit(1);
  }
  if (fclose(file) != 0) {
    fprintf(stderr, "iso: fail to close '%s'\n", attr_path);
    exit(1);
  }
  free(vert);
  free(tri);
  free(cells);
  free(attr);

  if ((file = fopen(xdmf_path, "w")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", xdmf_path);
    exit(1);
  }
  fprintf(file,
          "<Xdmf\n"
          "    Version=\"2\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Topology\n"
          "         TopologyType=\"Triangle\"\n"
          "         Dimensions=\"%ld\">\n"
          "        <DataItem\n"
          "            Dimensions=\"%ld 3\"\n"
          "            NumberType=\"Int\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Topology>\n"
          "      <Geometry>\n"
          "        <DataItem\n"
          "            Dimensions=\"%ld 3\"\n"
          "            Precision=\"4\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Geometry>\n"
          "      <Attribute\n"
          "          Name=\"u\">\n"
          "        <DataItem\n"
          "            Dimensions=\"%ld\"\n"
          "            Precision=\"4\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Attribute>\n"
          "    </Grid>\n"
          "  </Domain>\n"
          "</Xdmf>\n",
          ntri, ntri, tri_base, nvert, xyz_base, nvert, attr_base);
  if (fclose(file) != 0) {
    fprintf(stderr, "iso: fail to close '%s'\n", xdmf_path);
    exit(1);
  }
}
