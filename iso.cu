#include "table.inc"
#include <cuda.h>
#include <stdio.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

struct vec3i {
  __host__ __device__ vec3i() {}
  __host__ __device__ vec3i(int i) : x(i), y(i), z(i) {}
  __host__ __device__ vec3i(int x, int y, int z) : x(x), y(y), z(z) {}
  int x, y, z;
};

__host__ __device__ vec3i operator+(const vec3i &a, const vec3i &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ vec3i operator-(const vec3i &a, const vec3i &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__host__ __device__ vec3i operator*(const vec3i &a, const int b) {
  return {a.x * b, a.y * b, a.z * b};
}
__host__ vec3i min(const vec3i &a, const vec3i &b) {
  return vec3i(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}
__host__ vec3i max(const vec3i &a, const vec3i &b) {
  return vec3i(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}
__device__ __host__ vec3i operator>>(const vec3i v, const int s) {
  return vec3i(v.x >> s, v.y >> s, v.z >> s);
}

__device__ __host__ long leftShift3(long x) {
  x = (x | x << 32) & 0x1f00000000ffffull;
  x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full;
  x = (x | x << 4) & 0x10c30c30c30c30c3ull;
  x = (x | x << 2) & 0x1249249249249249ull;
  return x;
}

__device__ __host__ long mortonCode(const vec3i v) {
  return (leftShift3(uint32_t(v.z)) << 2) | (leftShift3(uint32_t(v.y)) << 1) |
         (leftShift3(uint32_t(v.x)) << 0);
}

__host__ __device__ bool operator==(const vec3i &a, const vec3i &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

struct vec3f {
  __host__ __device__ vec3f() {}
  __host__ __device__ vec3f(const float x, const float y, const float z)
      : x(x), y(y), z(z) {}
  __host__ __device__ vec3f(const float f) : x(f), y(f), z(f) {}
  __host__ __device__ vec3f(const vec3i o) : x(o.x), y(o.y), z(o.z) {}

  float x, y, z;
};

__host__ __device__ vec3f operator+(const vec3f &a, const vec3f &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ vec3f operator-(const vec3f &a, const vec3f &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__host__ __device__ vec3f operator*(const vec3f &a, const vec3f &b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
__host__ __device__ vec3f operator*(const vec3f &a, const float b) {
  return {a.x * b, a.y * b, a.z * b};
}
__host__ __device__ bool operator==(const vec3f &a, const vec3f &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

__host__ __device__ bool operator<(const vec3f &a, const vec3f &b) {
  return (a.x < b.x) ||
         ((a.x == b.x) && ((a.y < b.y) || (a.y == b.y) && (a.z < b.z)));
}

__host__ __device__ float4 operator+(const float4 &a, const float4 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
__host__ __device__ float4 operator*(const float4 &a, const float4 &b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
__host__ __device__ float4 operator-(const float4 &a, const float4 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
__host__ __device__ float4 operator*(const float4 &a, const float b) {
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}
__host__ __device__ float4 operator*(const float b, const float4 &a) {
  return {a.x * b, a.y * b, a.z * b, a.w * b};
}

__host__ __device__ bool operator==(const float4 &a, const float4 &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

struct CellCoords {
  __host__ __device__ CellCoords neighbor(const vec3i &delta) const {
    return {lower + delta * (1 << level), level};
  }
  __host__ __device__ vec3f center() const {
    return vec3f(lower) + vec3f(0.5f * (1 << level));
  }
  vec3i lower;
  int level;
};

__host__ __device__ bool operator<(const CellCoords &a, const CellCoords &b) {
  return (a.lower < b.lower) || (a.lower == b.lower && b.level < b.level);
}

__host__ __device__ bool operator==(const CellCoords &a, const CellCoords &b) {
  return (a.lower == b.lower) && (a.level == b.level);
}

struct Cell : public CellCoords {
  __device__ __host__ float4 asDualVertex() const {
    return make_float4(center().x, center().y, center().z, scalar);
  }
  float scalar;
};

__host__ __device__ bool operator==(const Cell &a, const Cell &b) {
  return ((const CellCoords &)a == (const CellCoords &)b) &&
         (a.scalar == b.scalar);
}

__host__ __device__ bool operator!=(const Cell &a, const Cell &b) {
  return !(a == b);
}

struct TriangleVertex {
  vec3f position;
  uint32_t triangleAndVertexID;
};

struct CompareByCoordsLowerOnly {
  __host__ __device__ CompareByCoordsLowerOnly(const vec3i coordOrigin)
      : coordOrigin(coordOrigin) {}
  __host__ __device__ bool operator()(const Cell &lhs,
                                      const CellCoords &rhs) const {
    return (mortonCode(lhs.lower - coordOrigin) <
            mortonCode(rhs.lower - coordOrigin));
  }
  const vec3i coordOrigin;
};

struct CompareVertices {
  __host__ __device__ bool operator()(const TriangleVertex &lhs,
                                      const TriangleVertex &rhs) const {
    const float4 a = (const float4 &)lhs;
    const float4 b = (const float4 &)rhs;

    return (const vec3f &)a < (const vec3f &)b;
  }
};

struct AMR {
  __host__ __device__ AMR(const vec3i coordOrigin,
                          const Cell *const __restrict__ cellArray,
                          const int numCells, const int maxLevel)
      : coordOrigin(coordOrigin), cellArray(cellArray), numCells(numCells),
        maxLevel(maxLevel) {}

  __host__ __device__ bool findActual(Cell &result, const CellCoords &coords) {
    const Cell *const __restrict__ begin = cellArray;
    const Cell *const __restrict__ end = cellArray + numCells;

    const Cell *it = thrust::system::detail::generic::scalar::lower_bound(
        begin, end, coords, CompareByCoordsLowerOnly(coordOrigin));

    if (it == end)
      return false;

    if ((it->lower >> it->level) == (coords.lower >> it->level) &&
        (it->level >= coords.level)) {
      result = *it;
      return true;
    }

    return false;
  }

  const Cell *const __restrict__ cellArray;
  const int numCells;
  const int maxLevel;
  const vec3i coordOrigin;
};

struct IsoExtractor {
  __device__ __host__ IsoExtractor(const float isoValue,
                                   TriangleVertex *outputArray,
                                   int outputArraySize, int *p_atomicCounter)
      : isoValue(isoValue), outputArray(outputArray),
        outputArraySize(outputArraySize), p_atomicCounter(p_atomicCounter) {}

  const float isoValue;
  TriangleVertex *const outputArray;
  const int outputArraySize;
  int *const p_atomicCounter;

  int __device__ allocTriangle() { return atomicAdd(p_atomicCounter, 1); }

  void __device__ doMarchingCubesOn(const vec3i mirror,
                                    const Cell zOrder[2][2][2]) {
    // we have OUR cells in z-order, but VTK case table assumes
    // everything is is VTK 'hexahedron' ordering, so let's rearrange
    // ... and while doing so, also make sure that we flip based on
    // which direction the parent cell created this dual from
    float4 vertex[8] = {
        zOrder[0 + mirror.z][0 + mirror.y][0 + mirror.x].asDualVertex(),
        zOrder[0 + mirror.z][0 + mirror.y][1 - mirror.x].asDualVertex(),
        zOrder[0 + mirror.z][1 - mirror.y][1 - mirror.x].asDualVertex(),
        zOrder[0 + mirror.z][1 - mirror.y][0 + mirror.x].asDualVertex(),
        zOrder[1 - mirror.z][0 + mirror.y][0 + mirror.x].asDualVertex(),
        zOrder[1 - mirror.z][0 + mirror.y][1 - mirror.x].asDualVertex(),
        zOrder[1 - mirror.z][1 - mirror.y][1 - mirror.x].asDualVertex(),
        zOrder[1 - mirror.z][1 - mirror.y][0 + mirror.x].asDualVertex()};

    int index = 0;
    for (int i = 0; i < 8; i++)
      if (vertex[i].w > isoValue)
        index += (1 << i);
    if (index == 0 || index == 0xff)
      return;

    for (const int8_t *edge = &vtkMarchingCubesTriangleCases[index][0];
         edge[0] > -1; edge += 3) {
      float4 triVertex[3];
      for (int ii = 0; ii < 3; ii++) {
        const int8_t *vert = vtkMarchingCubes_edges[edge[ii]];
        const float4 v0 = vertex[vert[0]];
        const float4 v1 = vertex[vert[1]];
        const float t = (isoValue - v0.w) / float(v1.w - v0.w);
        triVertex[ii] = (1.f - t) * v0 + t * v1;
      }

      if (triVertex[1] == triVertex[0])
        continue;
      if (triVertex[2] == triVertex[0])
        continue;
      if (triVertex[1] == triVertex[2])
        continue;

      const int triangleID = allocTriangle();
      if (triangleID >= 3 * outputArraySize)
        continue;

      for (int j = 0; j < 3; j++) {
        (int &)triVertex[j].w = (4 * triangleID + j);
        (float4 &)outputArray[3 * triangleID + j] = triVertex[j];
      }
    }
  }
};

__global__ void extractTriangles(const vec3i coordOrigin,
                                 const Cell *const __restrict__ cellArray,
                                 const int numCells, const int maxLevel,
                                 const float isoValue,
                                 TriangleVertex *__restrict__ outVertex,
                                 const int outVertexSize,
                                 int *p_numGeneratedTriangles) {
  AMR amr(coordOrigin, cellArray, numCells, maxLevel);

  const size_t threadID = threadIdx.x + size_t(blockDim.x) * blockIdx.x;

  const int workID = threadID / 8;
  if (workID >= numCells)
    return;
  const int directionID = threadID % 8;
  const Cell currentCell = cellArray[workID];

  const int dz = (directionID & 4) ? 1 : -1;
  const int dy = (directionID & 2) ? 1 : -1;
  const int dx = (directionID & 1) ? 1 : -1;

  Cell corner[2][2][2];
  for (int iz = 0; iz < 2; iz++)
    for (int iy = 0; iy < 2; iy++)
      for (int ix = 0; ix < 2; ix++) {
        const vec3i delta = vec3i(dx * ix, dy * iy, dz * iz);
        const CellCoords cornerCoords = currentCell.neighbor(delta);

        if (!amr.findActual(corner[iz][iy][ix], cornerCoords))
          // corner does not exist - currentcell is on a boundary, and
          // this is not a dual cell
          return;

        if (corner[iz][iy][ix].level < currentCell.level)
          // somebody else will generate this same cell from a finer
          // level...
          return;

        if (corner[iz][iy][ix].level == currentCell.level &&
            corner[iz][iy][ix] < currentCell)
          // this other cell will generate this dual cell...
          return;
      }

  IsoExtractor isoExtractor(isoValue, outVertex, outVertexSize,
                            p_numGeneratedTriangles);
  isoExtractor.doMarchingCubesOn({dx == -1, dy == -1, dz == -1}, corner);
}

__global__ void
createVertexArray(int *p_atomicCounter,
                  const TriangleVertex *const __restrict__ vertices,
                  int numVertices, float3 *outVertexArray,
                  int outVertexArraySize, int3 *outIndexArray) {
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadID >= numVertices)
    return;

  const TriangleVertex vertex = vertices[threadID];
  if (threadID > 0 && vertex.position == vertices[threadID - 1].position)
    // not unique...
    return;

  int vertexArrayID = atomicAdd(p_atomicCounter, 1);
  if (vertexArrayID >= outVertexArraySize)
    return;

  outVertexArray[vertexArrayID] = (float3 &)vertex.position;

  for (int i = threadID;
       i < numVertices && vertices[i].position == vertex.position; i++) {
    int triangleAndVertexID = vertices[i].triangleAndVertexID;
    int targetVertexID = triangleAndVertexID % 4;
    int targetTriangleID = triangleAndVertexID / 4;
    int *triIndices = &outIndexArray[targetTriangleID].x;
    triIndices[targetVertexID] = vertexArrayID;
  }
}

static vec3i coordOrigin;
static int comp(const void *av, const void *bv) {
  struct Cell *a, *b;
  a = (struct Cell *)av;
  b = (struct Cell *)bv;
  return mortonCode(a->lower - coordOrigin) -
         mortonCode(b->lower - coordOrigin);
}

int main(int argc, char **argv) {
  float isoValue, *attr;
  float3 *vert;
  int3 *tri;
  int maxLevel;
  long i, j, nvert, ntri, numCells, size;
  FILE *file, *cell_file, *scalar_file;
  int cell[4];
  char attr_path[FILENAME_MAX], xyz_path[FILENAME_MAX], tri_path[FILENAME_MAX],
      xdmf_path[FILENAME_MAX], *attr_base, *xyz_base, *tri_base, *cell_path,
      *scalar_path, *output_path;
  struct Cell *cells;
  if (argc != 5) {
    fprintf(stderr, "iso in.cells in.scalars isoValue mesh\n");
    exit(1);
  }
  cell_path = argv[1];
  scalar_path = argv[2];
  isoValue = std::stof(argv[3]);
  output_path = argv[4];
  coordOrigin = 1 << 30;
  vec3i bounds_lower(1 << 30);
  vec3i bounds_upper(-(1 << 30));
  maxLevel = 0;
  numCells = 0;
  if ((cell_file = fopen(cell_path, "r")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", cell_path);
    exit(1);
  }
  if ((scalar_file = fopen(scalar_path, "r")) == NULL) {
    fprintf(stderr, "iso: error: fail to open '%s'\n", scalar_path);
    exit(1);
  }
  fseek(cell_file, 0, SEEK_END);
  size = ftell(cell_file);
  fseek(cell_file, 0, SEEK_SET);
  numCells = size / (4 * sizeof(int));
  fprintf(stderr, "%ld\n", numCells);
  if ((cells = (Cell *)malloc(numCells * sizeof *cells)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  for (i = 0; i < numCells; i++) {
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
    maxLevel = std::max(maxLevel, cells[i].level);
    bounds_lower = min(bounds_lower, cells[i].lower);
    bounds_upper =
        max(bounds_upper, cells[i].lower + vec3i(1 << cells[i].level));
    coordOrigin = min(coordOrigin, cells[i].lower);
  }
  if (fclose(cell_file) != 0) {
    fprintf(stderr, "cylinder: error: fail to close '%s'\n", cell_path);
    exit(1);
  }
  if (fclose(scalar_file) != 0) {
    fprintf(stderr, "cylinder: error: fail to close '%s'\n", scalar_path);
    exit(1);
  }
  coordOrigin.x &= ~((1 << maxLevel) - 1);
  coordOrigin.y &= ~((1 << maxLevel) - 1);
  coordOrigin.z &= ~((1 << maxLevel) - 1);
  qsort(cells, numCells, sizeof *cells, comp);
  thrust::device_vector<Cell> d_cells(numCells);
  thrust::copy(cells, cells + numCells, d_cells.begin());
  cudaDeviceSynchronize();
  thrust::device_vector<int> d_atomicCounter(1);
  thrust::device_vector<TriangleVertex> d_triangleVertices(0);
  {
    d_atomicCounter[0] = 0;
    size_t numJobs = 8 * numCells;
    int blockSize = 512;
    int numBlocks = (numJobs + blockSize - 1) / blockSize;
    extractTriangles<<<numBlocks, blockSize>>>(
        coordOrigin, thrust::raw_pointer_cast(d_cells.data()), d_cells.size(),
        maxLevel, isoValue, thrust::raw_pointer_cast(d_triangleVertices.data()),
        d_triangleVertices.size(),
        thrust::raw_pointer_cast(d_atomicCounter.data()));
  }
  cudaDeviceSynchronize();
  int numTriangles = d_atomicCounter[0];
  d_triangleVertices.resize(3 * numTriangles);

  {
    d_atomicCounter[0] = 0;
    size_t numJobs = 8 * numCells;
    int blockSize = 512;
    int numBlocks = (numJobs + blockSize - 1) / blockSize;
    extractTriangles<<<numBlocks, // dim3(1024,divUp(numBlocks,1024)),
                       blockSize>>>(
        coordOrigin, thrust::raw_pointer_cast(d_cells.data()), d_cells.size(),
        maxLevel, isoValue, thrust::raw_pointer_cast(d_triangleVertices.data()),
        d_triangleVertices.size(),
        thrust::raw_pointer_cast(d_atomicCounter.data()));
  }
  cudaDeviceSynchronize();
  thrust::sort(d_triangleVertices.begin(), d_triangleVertices.end(),
               CompareVertices());
  cudaDeviceSynchronize();
  thrust::device_vector<float3> d_vertexArray(0);
  thrust::device_vector<int3> d_indexArray(numTriangles);
  {
    d_atomicCounter[0] = 0;
    int numJobs = 3 * numTriangles;
    int blockSize = 512;
    int numBlocks = (numJobs + blockSize - 1) / blockSize;
    createVertexArray<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_atomicCounter.data()),
        thrust::raw_pointer_cast(d_triangleVertices.data()),
        d_triangleVertices.size(),
        thrust::raw_pointer_cast(d_vertexArray.data()), d_vertexArray.size(),
        thrust::raw_pointer_cast(d_indexArray.data()));
  }
  cudaDeviceSynchronize();
  int numVertices = d_atomicCounter[0];
  d_vertexArray.resize(numVertices);
  {
    d_atomicCounter[0] = 0;
    int numJobs = 3 * numTriangles;
    int blockSize = 512;
    int numBlocks = (numJobs + blockSize - 1) / blockSize;
    createVertexArray<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_atomicCounter.data()),
        thrust::raw_pointer_cast(d_triangleVertices.data()),
        d_triangleVertices.size(),
        thrust::raw_pointer_cast(d_vertexArray.data()), d_vertexArray.size(),
        thrust::raw_pointer_cast(d_indexArray.data()));
  }
  cudaDeviceSynchronize();
  ntri = d_indexArray.size();
  nvert = d_vertexArray.size();
  if ((vert = (float3 *)malloc(nvert * sizeof *vert)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  if ((tri = (int3 *)malloc(ntri * sizeof *tri)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  thrust::copy(d_vertexArray.begin(), d_vertexArray.end(), vert);
  thrust::copy(d_indexArray.begin(), d_indexArray.end(), tri);

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
  if (fwrite(vert, nvert * sizeof *vert, 1, file) != 1) {
    fprintf(stderr, "iso: error: fail to write '%s'\n", xyz_path);
    exit(1);
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

  for (j = 0; j < nvert; j++)
    attr[j] = vert[j].z;

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
          "      <Attribute>\n"
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
