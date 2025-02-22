#include "table.inc"
#include <cuda.h>
#include <stdio.h>
#include <thrust/binary_search.h>

struct vec3i {
  int x, y, z;
};

static __device__ vec3i operator>>(const vec3i v, const int s) {
  vec3i u;
  u.x = v.x >> s;
  u.y = v.y >> s;
  u.z = v.z >> s;
  return u;
}

static __device__ __host__ long leftShift3(long x) {
  x = (x | x << 32) & 0x1f00000000ffffull;
  x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full;
  x = (x | x << 4) & 0x10c30c30c30c30c3ull;
  x = (x | x << 2) & 0x1249249249249249ull;
  return x;
}

static __device__ __host__ long morton(int x, int y, int z) {
  return (leftShift3(uint32_t(z)) << 2) | (leftShift3(uint32_t(y)) << 1) |
         (leftShift3(uint32_t(x)) << 0);
}

struct vec3f {
  __device__ vec3f() {}
  __device__ vec3f(const vec3i o) : x(o.x), y(o.y), z(o.z) {}
  float x, y, z;
};

struct Vertex {
  vec3f position;
  float scalar, field;
  uint32_t id;
};

static __device__ bool operator==(const vec3f &a, const vec3f &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
static __device__ __host__ bool operator<(const vec3f &a, const vec3f &b) {
  return (a.x < b.x) ||
         ((a.x == b.x) && ((a.y < b.y) || (a.y == b.y) && (a.z < b.z)));
}

struct Cell {
  vec3i lower;
  int level;
  uint64_t morton;
  float scalar, field;
};

static __device__ struct Vertex dual(struct Cell c) {
  struct Vertex v;
  v.position.x = c.lower.x + 0.5 * (1 << c.level);
  v.position.y = c.lower.y + 0.5 * (1 << c.level);
  v.position.z = c.lower.z + 0.5 * (1 << c.level);
  v.scalar = c.scalar;
  v.field = c.field;
  return v;
}

struct CompareMorton {
  __device__ bool operator()(const Cell &a, const uint64_t b) {
    return a.morton < b;
  }
};

struct AMR {
  __device__ AMR(Cell *cells, unsigned long long ncell)
      : cells(cells), ncell(ncell) {}

  __device__ bool findActual(struct Cell *result, const vec3i lower,
                             int level) {
    int f;
    const Cell *it = thrust::system::detail::generic::scalar::lower_bound(
        cells, cells + ncell, morton(lower.x, lower.y, lower.z),
        CompareMorton());
    if (it == cells + ncell)
      return false;
    *result = *it;

    f = max(level, result->level);
    if ((result->lower >> f) == (lower >> f))
      return true;
    if (it > cells) {
      *result = it[-1];
      f = max(level, result->level);
      if ((result->lower >> f) == (lower >> f))
        return true;
    }
    return false;
  };

  Cell *cells;
  unsigned long long ncell;
};

__global__ void extract(Cell *cells, unsigned long long ncell, int maxlevel,
                        float iso, Vertex *out, int size,
                        unsigned long long *cnt) {
  size_t tid;
  int x, y, z, index, i, j, k, ii, wid, did, dx, dy, dz, ix, iy, iz;
  unsigned long long id;
  int8_t *edge, *vert;
  float t;
  Vertex v0, v1, triVertex[3];
  vec3i lower;
  Cell corner[2][2][2], cell;
  AMR amr(cells, ncell);
  tid = threadIdx.x + size_t(blockDim.x) * blockIdx.x;
  wid = tid / 8;
  if (wid >= ncell)
    return;
  did = tid % 8;
  cell = cells[wid];
  dz = (did & 4) ? 1 : -1;
  dy = (did & 2) ? 1 : -1;
  dx = (did & 1) ? 1 : -1;
  for (iz = 0; iz < 2; iz++)
    for (iy = 0; iy < 2; iy++)
      for (ix = 0; ix < 2; ix++) {
        lower.x = cell.lower.x + dx * ix * (1 << cell.level);
        lower.y = cell.lower.y + dy * iy * (1 << cell.level);
        lower.z = cell.lower.z + dz * iz * (1 << cell.level);
        if (!amr.findActual(&corner[iz][iy][ix], lower, cell.level))
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
  Vertex vertex[8] = {
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
    id = atomicAdd(cnt, 1ull);
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

__global__ void createVertexArray(unsigned long long *cnt,
                                  const Vertex *const __restrict__ vertices,
                                  int nvert, Vertex *vert, int size,
                                  int3 *index) {
  int i, j, k, l, id, tid, *tri;
  Vertex vertex;
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

static int comp_vert(const void *av, const void *bv) {
  struct Vertex *a, *b;
  a = (struct Vertex *)av;
  b = (struct Vertex *)bv;
  return a->position < b->position;
}

int main(int argc, char **argv) {
  double X0, Y0, Z0, L;
  float iso, *attr, xyz[3];
  int3 *tri, *d_tri;
  size_t numJobs;
  int Verbose, maxlevel, numBlocks;
  long j, size;
  FILE *file, *cell_file, *scalar_file, *field_file;
  int cell[4], ox, oy, oz, minlevel, ScaleFlag;
  char attr_path[FILENAME_MAX], xyz_path[FILENAME_MAX], tri_path[FILENAME_MAX],
      xdmf_path[FILENAME_MAX], *attr_base, *xyz_base, *tri_base, *cell_path,
      *scalar_path, *field_path, *output_path, *end;
  struct Cell *cells, *d_cells;
  struct Vertex *d_tv, *tv, *d_vert, *vert;
  unsigned long long nvert, ntri, ncell, *d_cnt, i;
  cudaError_t code;
  enum { blockSize = 512 };

  Verbose = 0;
  ScaleFlag = 0;
  while (*++argv != NULL && argv[0][0] == '-')
    switch (argv[0][1]) {
    case 'h':
      fprintf(
          stderr,
          "Usage: iso [-v] [-s X0 Y0 Z0 L minlevel] in.cells in.scalar "
          "in.field iso mesh\n\n"
          "Example:\n"
          "  iso -v data.cells data.scalar data.field 0.5 output\n\n"
          "Arguments:\n"
          "  in.cells   Binary file describing the AMR cell structure.\n"
          "  in.scalar  Binary file with scalar field values.\n"
          "  in.field   Binary file with additional field values.\n"
          "  iso        Iso-surface value to extract (e.g., 0.5).\n"
          "  mesh       Output file name prefix for generated mesh.\n\n"
          "Options:\n"
          "  -s         Domain center, size, and minimum level for rescaling\n"
          "  -v         Enable verbose output.\n"
          "  -h         Show this help message and exit.\n");
      exit(1);
    case 's':
      argv++;
      if (argv[0] == NULL || argv[1] == NULL || argv[2] == NULL ||
          argv[3] == NULL || argv[3] == NULL) {
        fprintf(stderr, "iso: error: -w needs five arguments\n");
        exit(1);
      }
      ScaleFlag = 1;
      X0 = strtod(*argv, &end);
      if (*end != '\0') {
        fprintf(stderr, "iso: error: '%s' is not a double\n", *argv);
        exit(1);
      }
      argv++;
      Y0 = strtod(*argv, &end);
      if (*end != '\0') {
        fprintf(stderr, "iso: error: '%s' is not a double\n", *argv);
        exit(1);
      }
      argv++;
      Z0 = strtod(*argv, &end);
      if (*end != '\0') {
        fprintf(stderr, "iso: error: '%s' is not a double\n", *argv);
        exit(1);
      }
      argv++;
      L = strtod(*argv, &end);
      if (*end != '\0') {
        fprintf(stderr, "iso: error: '%s' is not a double\n", *argv);
        exit(1);
      }
      argv++;
      minlevel = strtol(*argv, &end, 10);
      if (*end != '\0') {
        fprintf(stderr, "iso: error: '%s' is not an integer\n", *argv);
        exit(1);
      }
      break;
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
        morton(cells[i].lower.x, cells[i].lower.y, cells[i].lower.z);
  }

  if (Verbose)
    fprintf(stderr, "iso: ncell, maxlevel, origin: %llu %d [%d %d %d]\n", ncell,
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
  cudaMalloc(&d_cells, ncell * sizeof *d_cells);
  cudaMemcpy(d_cells, cells, ncell * sizeof *d_cells, cudaMemcpyHostToDevice);
  cudaMalloc(&d_cnt, sizeof *d_cnt);
  cudaMemset(d_cnt, 0, sizeof *d_cnt);
  numJobs = 8 * ncell;
  numBlocks = (numJobs + blockSize - 1) / blockSize;
  extract<<<numBlocks, blockSize>>>(d_cells, ncell, maxlevel, iso, NULL, 0,
                                    d_cnt);
  cudaDeviceSynchronize();
  if ((code = cudaPeekAtLastError()) != cudaSuccess) {
    fprintf(stderr, "iso: error: %s\n", cudaGetErrorString(code));
    exit(1);
  }
  cudaMemcpy(&ntri, d_cnt, sizeof *d_cnt, cudaMemcpyDeviceToHost);
  if (Verbose)
    fprintf(stderr, "iso: ntri: %llu\n", ntri);
  if (ntri == 0) {
    fprintf(stderr, "iso: error: no triangles in the mesh\n");
    exit(1);
  }
  cudaMalloc(&d_tv, 3 * ntri * sizeof *d_tv);
  cudaMemset(d_cnt, 0, sizeof *d_cnt);
  extract<<<numBlocks, blockSize>>>(d_cells, ncell, maxlevel, iso, d_tv,
                                    3 * ntri, d_cnt);
  cudaDeviceSynchronize();
  cudaFree(d_cells);
  if ((tv = (struct Vertex *)malloc(3 * ntri * sizeof *tv)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  cudaMemcpy(tv, d_tv, 3 * ntri * sizeof *tv, cudaMemcpyDeviceToHost);
  qsort(tv, 3 * ntri, sizeof *tv, comp_vert);
  cudaMemcpy(d_tv, tv, 3 * ntri * sizeof *tv, cudaMemcpyHostToDevice);
  free(tv);
  cudaMemset(d_cnt, 0, sizeof *d_cnt);
  numJobs = 3 * ntri;
  numBlocks = (numJobs + blockSize - 1) / blockSize;
  createVertexArray<<<numBlocks, blockSize>>>(d_cnt, d_tv, 3 * ntri, NULL, 0,
                                              NULL);
  cudaDeviceSynchronize();
  cudaMemcpy(&nvert, d_cnt, sizeof *d_cnt, cudaMemcpyDeviceToHost);
  if (Verbose)
    fprintf(stderr, "iso: nvert: %llu\n", nvert);
  cudaMalloc(&d_tri, ntri * sizeof *d_tri);
  cudaMalloc(&d_vert, nvert * sizeof *d_vert);
  cudaMemset(d_cnt, 0, sizeof *d_cnt);
  createVertexArray<<<numBlocks, blockSize>>>(d_cnt, d_tv, 3 * ntri, d_vert,
                                              nvert, d_tri);
  cudaDeviceSynchronize();
  if ((vert = (Vertex *)malloc(nvert * sizeof *vert)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  if ((tri = (int3 *)malloc(ntri * sizeof *tri)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  cudaMemcpy(vert, d_vert, nvert * sizeof *d_vert, cudaMemcpyDeviceToHost);
  cudaMemcpy(tri, d_tri, ntri * sizeof *d_tri, cudaMemcpyDeviceToHost);
  cudaFree(d_tri);
  cudaFree(d_vert);
  cudaFree(d_tv);
  cudaFree(d_cnt);

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
  if (ScaleFlag == 0) {
    for (i = 0; i < nvert; i++) {
      xyz[0] = vert[i].position.x;
      xyz[1] = vert[i].position.y;
      xyz[2] = vert[i].position.z;
      if (fwrite(xyz, sizeof xyz, 1, file) != 1) {
        fprintf(stderr, "iso: error: fail to write '%s'\n", xyz_path);
        exit(1);
      }
    }
  } else {
    double h;
    h = L / (1 << minlevel);
    if (Verbose)
      fprintf(stderr, "iso: X0 Y0 Z0 L minlevel: [%g %g %g] %g %d\n", X0, Y0,
              Z0, L, minlevel);
    for (i = 0; i < nvert; i++) {
      xyz[0] = vert[i].position.x * h + X0;
      xyz[1] = vert[i].position.y * h + Y0;
      xyz[2] = vert[i].position.z * h + Z0;
      if (fwrite(xyz, sizeof xyz, 1, file) != 1) {
        fprintf(stderr, "iso: error: fail to write '%s'\n", xyz_path);
        exit(1);
      }
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
          "         Dimensions=\"%llu\">\n"
          "        <DataItem\n"
          "            Dimensions=\"%llu 3\"\n"
          "            NumberType=\"Int\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Topology>\n"
          "      <Geometry>\n"
          "        <DataItem\n"
          "            Dimensions=\"%llu 3\"\n"
          "            Precision=\"4\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Geometry>\n"
          "      <Attribute\n"
          "          Center=\"Node\"\n"
          "          Name=\"u\">\n"
          "        <DataItem\n"
          "            Dimensions=\"%llu\"\n"
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
    fprintf(stderr, "iso: error: fail to close '%s'\n", xdmf_path);
    exit(1);
  }
}
