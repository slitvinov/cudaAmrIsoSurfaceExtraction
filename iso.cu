#include <cuda.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
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

__constant__ int8_t vtkMarchingCubesTriangleCases[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 0 0 */
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 1 1 */
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 2 1 */
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       /* 3 2 */
    {1, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},   /* 4 1 */
    {0, 3, 8, 1, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      /* 5 3 */
    {9, 11, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      /* 6 2 */
    {2, 3, 8, 2, 8, 11, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},        /* 7 5 */
    {3, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},   /* 8 1 */
    {0, 2, 10, 8, 0, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 9 2 */
    {1, 0, 9, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      /* 10 3 */
    {1, 2, 10, 1, 10, 9, 9, 10, 8, -1, -1, -1, -1, -1, -1, -1},       /* 11 5 */
    {3, 1, 11, 10, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 12 2 */
    {0, 1, 11, 0, 11, 8, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},      /* 13 5 */
    {3, 0, 9, 3, 9, 10, 10, 9, 11, -1, -1, -1, -1, -1, -1, -1},       /* 14 5 */
    {9, 11, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 15 8 */
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 16 1 */
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       /* 17 2 */
    {0, 9, 1, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       /* 18 3 */
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},          /* 19 5 */
    {1, 11, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      /* 20 4 */
    {3, 7, 4, 3, 4, 0, 1, 11, 2, -1, -1, -1, -1, -1, -1, -1},         /* 21 7 */
    {9, 11, 2, 9, 2, 0, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1},         /* 22 7 */
    {2, 9, 11, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},          /* 23 14 */
    {8, 7, 4, 3, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 24 3 */
    {10, 7, 4, 10, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},      /* 25 5 */
    {9, 1, 0, 8, 7, 4, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1},       /* 26 6 */
    {4, 10, 7, 9, 10, 4, 9, 2, 10, 9, 1, 2, -1, -1, -1, -1},        /* 27 9 */
    {3, 1, 11, 3, 11, 10, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1},     /* 28 7 */
    {1, 11, 10, 1, 10, 4, 1, 4, 0, 7, 4, 10, -1, -1, -1, -1},       /* 29 11 */
    {4, 8, 7, 9, 10, 0, 9, 11, 10, 10, 3, 0, -1, -1, -1, -1},       /* 30 12 */
    {4, 10, 7, 4, 9, 10, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},    /* 31 5 */
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 32 1 */
    {9, 4, 5, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 33 3 */
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 34 2 */
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},        /* 35 5 */
    {1, 11, 2, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 36 3 */
    {3, 8, 0, 1, 11, 2, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1},       /* 37 6 */
    {5, 11, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},       /* 38 5 */
    {2, 5, 11, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},          /* 39 9 */
    {9, 4, 5, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 40 4 */
    {0, 2, 10, 0, 10, 8, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1},      /* 41 7 */
    {0, 4, 5, 0, 5, 1, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1},       /* 42 7 */
    {2, 5, 1, 2, 8, 5, 2, 10, 8, 4, 5, 8, -1, -1, -1, -1},          /* 43 11 */
    {11, 10, 3, 11, 3, 1, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1},     /* 44 7 */
    {4, 5, 9, 0, 1, 8, 8, 1, 11, 8, 11, 10, -1, -1, -1, -1},        /* 45 12 */
    {5, 0, 4, 5, 10, 0, 5, 11, 10, 10, 3, 0, -1, -1, -1, -1},       /* 46 14 */
    {5, 8, 4, 5, 11, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1},     /* 47 5 */
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 48 2 */
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},        /* 49 5 */
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},        /* 50 5 */
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 51 8 */
    {9, 8, 7, 9, 7, 5, 11, 2, 1, -1, -1, -1, -1, -1, -1, -1},       /* 52 7 */
    {11, 2, 1, 9, 0, 5, 5, 0, 3, 5, 3, 7, -1, -1, -1, -1},          /* 53 12 */
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 11, 2, 5, -1, -1, -1, -1},          /* 54 11 */
    {2, 5, 11, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},       /* 55 5 */
    {7, 5, 9, 7, 9, 8, 3, 2, 10, -1, -1, -1, -1, -1, -1, -1},       /* 56 7 */
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 10, 7, -1, -1, -1, -1},          /* 57 14 */
    {2, 10, 3, 0, 8, 1, 1, 8, 7, 1, 7, 5, -1, -1, -1, -1},          /* 58 12 */
    {10, 1, 2, 10, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},      /* 59 5 */
    {9, 8, 5, 8, 7, 5, 11, 3, 1, 11, 10, 3, -1, -1, -1, -1},        /* 60 10 */
    {5, 0, 7, 5, 9, 0, 7, 0, 10, 1, 11, 0, 10, 0, 11, -1},          /* 61 7 */
    {10, 0, 11, 10, 3, 0, 11, 0, 5, 8, 7, 0, 5, 0, 7, -1},          /* 62 7 */
    {10, 5, 11, 7, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 63 2 */
    {11, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 64 1 */
    {0, 3, 8, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 65 4 */
    {9, 1, 0, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 66 3 */
    {1, 3, 8, 1, 8, 9, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1},       /* 67 7 */
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 68 2 */
    {1, 5, 6, 1, 6, 2, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1},        /* 69 7 */
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},        /* 70 5 */
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},           /* 71 11 */
    {2, 10, 3, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},   /* 72 3 */
    {10, 8, 0, 10, 0, 2, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1},     /* 73 7 */
    {0, 9, 1, 2, 10, 3, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1},      /* 74 6 */
    {5, 6, 11, 1, 2, 9, 9, 2, 10, 9, 10, 8, -1, -1, -1, -1},        /* 75 12 */
    {6, 10, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},       /* 76 5 */
    {0, 10, 8, 0, 5, 10, 0, 1, 5, 5, 6, 10, -1, -1, -1, -1},        /* 77 14 */
    {3, 6, 10, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},          /* 78 9 */
    {6, 9, 5, 6, 10, 9, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},      /* 79 5 */
    {5, 6, 11, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 80 3 */
    {4, 0, 3, 4, 3, 7, 6, 11, 5, -1, -1, -1, -1, -1, -1, -1},       /* 81 7 */
    {1, 0, 9, 5, 6, 11, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1},       /* 82 6 */
    {11, 5, 6, 1, 7, 9, 1, 3, 7, 7, 4, 9, -1, -1, -1, -1},          /* 83 12 */
    {6, 2, 1, 6, 1, 5, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1},        /* 84 7 */
    {1, 5, 2, 5, 6, 2, 3, 4, 0, 3, 7, 4, -1, -1, -1, -1},           /* 85 10 */
    {8, 7, 4, 9, 5, 0, 0, 5, 6, 0, 6, 2, -1, -1, -1, -1},           /* 86 12 */
    {7, 9, 3, 7, 4, 9, 3, 9, 2, 5, 6, 9, 2, 9, 6, -1},              /* 87 7 */
    {3, 2, 10, 7, 4, 8, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1},      /* 88 6 */
    {5, 6, 11, 4, 2, 7, 4, 0, 2, 2, 10, 7, -1, -1, -1, -1},         /* 89 12 */
    {0, 9, 1, 4, 8, 7, 2, 10, 3, 5, 6, 11, -1, -1, -1, -1},         /* 90 13 */
    {9, 1, 2, 9, 2, 10, 9, 10, 4, 7, 4, 10, 5, 6, 11, -1},          /* 91 6 */
    {8, 7, 4, 3, 5, 10, 3, 1, 5, 5, 6, 10, -1, -1, -1, -1},         /* 92 12 */
    {5, 10, 1, 5, 6, 10, 1, 10, 0, 7, 4, 10, 0, 10, 4, -1},         /* 93 7 */
    {0, 9, 5, 0, 5, 6, 0, 6, 3, 10, 3, 6, 8, 7, 4, -1},             /* 94 6 */
    {6, 9, 5, 6, 10, 9, 4, 9, 7, 7, 9, 10, -1, -1, -1, -1},         /* 95 3 */
    {11, 9, 4, 6, 11, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},   /* 96 2 */
    {4, 6, 11, 4, 11, 9, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1},      /* 97 7 */
    {11, 1, 0, 11, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},      /* 98 5 */
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 11, 1, -1, -1, -1, -1},          /* 99 14 */
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},        /* 100 5 */
    {3, 8, 0, 1, 9, 2, 2, 9, 4, 2, 4, 6, -1, -1, -1, -1},           /* 101 12 */
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 102 8 */
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},        /* 103 5 */
    {11, 9, 4, 11, 4, 6, 10, 3, 2, -1, -1, -1, -1, -1, -1, -1},     /* 104 7 */
    {0, 2, 8, 2, 10, 8, 4, 11, 9, 4, 6, 11, -1, -1, -1, -1},        /* 105 10 */
    {3, 2, 10, 0, 6, 1, 0, 4, 6, 6, 11, 1, -1, -1, -1, -1},         /* 106 12 */
    {6, 1, 4, 6, 11, 1, 4, 1, 8, 2, 10, 1, 8, 1, 10, -1},           /* 107 7 */
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 10, 3, 6, -1, -1, -1, -1},          /* 108 11 */
    {8, 1, 10, 8, 0, 1, 10, 1, 6, 9, 4, 1, 6, 1, 4, -1},            /* 109 7 */
    {3, 6, 10, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},       /* 110 5 */
    {6, 8, 4, 10, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 111 2 */
    {7, 6, 11, 7, 11, 8, 8, 11, 9, -1, -1, -1, -1, -1, -1, -1},     /* 112 5 */
    {0, 3, 7, 0, 7, 11, 0, 11, 9, 6, 11, 7, -1, -1, -1, -1},        /* 113 11 */
    {11, 7, 6, 1, 7, 11, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},         /* 114 9 */
    {11, 7, 6, 11, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},      /* 115 5 */
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},           /* 116 14 */
    {2, 9, 6, 2, 1, 9, 6, 9, 7, 0, 3, 9, 7, 9, 3, -1},              /* 117 7 */
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},        /* 118 5 */
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 119 2 */
    {2, 10, 3, 11, 8, 6, 11, 9, 8, 8, 7, 6, -1, -1, -1, -1},        /* 120 12 */
    {2, 7, 0, 2, 10, 7, 0, 7, 9, 6, 11, 7, 9, 7, 11, -1},           /* 121 7 */
    {1, 0, 8, 1, 8, 7, 1, 7, 11, 6, 11, 7, 2, 10, 3, -1},           /* 122 6 */
    {10, 1, 2, 10, 7, 1, 11, 1, 6, 6, 1, 7, -1, -1, -1, -1},        /* 123 3 */
    {8, 6, 9, 8, 7, 6, 9, 6, 1, 10, 3, 6, 1, 6, 3, -1},             /* 124 7 */
    {0, 1, 9, 10, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 125 4 */
    {7, 0, 8, 7, 6, 0, 3, 0, 10, 10, 0, 6, -1, -1, -1, -1},         /* 126 3 */
    {7, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 127 1 */
    {7, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 128 1 */
    {3, 8, 0, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 129 3 */
    {0, 9, 1, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 130 4 */
    {8, 9, 1, 8, 1, 3, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1},       /* 131 7 */
    {11, 2, 1, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},   /* 132 3 */
    {1, 11, 2, 3, 8, 0, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1},      /* 133 6 */
    {2, 0, 9, 2, 9, 11, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1},      /* 134 7 */
    {6, 7, 10, 2, 3, 11, 11, 3, 8, 11, 8, 9, -1, -1, -1, -1},       /* 135 12 */
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 136 2 */
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},        /* 137 5 */
    {2, 6, 7, 2, 7, 3, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1},        /* 138 7 */
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},           /* 139 14 */
    {11, 6, 7, 11, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},      /* 140 5 */
    {11, 6, 7, 1, 11, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},         /* 141 9 */
    {0, 7, 3, 0, 11, 7, 0, 9, 11, 6, 7, 11, -1, -1, -1, -1},        /* 142 11 */
    {7, 11, 6, 7, 8, 11, 8, 9, 11, -1, -1, -1, -1, -1, -1, -1},     /* 143 5 */
    {6, 4, 8, 10, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 144 2 */
    {3, 10, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},       /* 145 5 */
    {8, 10, 6, 8, 6, 4, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1},       /* 146 7 */
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 10, 6, 3, -1, -1, -1, -1},          /* 147 11 */
    {6, 4, 8, 6, 8, 10, 2, 1, 11, -1, -1, -1, -1, -1, -1, -1},      /* 148 7 */
    {1, 11, 2, 3, 10, 0, 0, 10, 6, 0, 6, 4, -1, -1, -1, -1},        /* 149 12 */
    {4, 8, 10, 4, 10, 6, 0, 9, 2, 2, 9, 11, -1, -1, -1, -1},        /* 150 10 */
    {11, 3, 9, 11, 2, 3, 9, 3, 4, 10, 6, 3, 4, 3, 6, -1},           /* 151 7 */
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},        /* 152 5 */
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 153 8 */
    {1, 0, 9, 2, 4, 3, 2, 6, 4, 4, 8, 3, -1, -1, -1, -1},           /* 154 12 */
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},        /* 155 5 */
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 11, -1, -1, -1, -1},          /* 156 14 */
    {11, 0, 1, 11, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},      /* 157 5 */
    {4, 3, 6, 4, 8, 3, 6, 3, 11, 0, 9, 3, 11, 3, 9, -1},            /* 158 7 */
    {11, 4, 9, 6, 4, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},   /* 159 2 */
    {4, 5, 9, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 160 3 */
    {0, 3, 8, 4, 5, 9, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1},       /* 161 6 */
    {5, 1, 0, 5, 0, 4, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1},       /* 162 7 */
    {10, 6, 7, 8, 4, 3, 3, 4, 5, 3, 5, 1, -1, -1, -1, -1},          /* 163 12 */
    {9, 4, 5, 11, 2, 1, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1},      /* 164 6 */
    {6, 7, 10, 1, 11, 2, 0, 3, 8, 4, 5, 9, -1, -1, -1, -1},         /* 165 13 */
    {7, 10, 6, 5, 11, 4, 4, 11, 2, 4, 2, 0, -1, -1, -1, -1},        /* 166 12 */
    {3, 8, 4, 3, 4, 5, 3, 5, 2, 11, 2, 5, 10, 6, 7, -1},            /* 167 6 */
    {7, 3, 2, 7, 2, 6, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1},        /* 168 7 */
    {9, 4, 5, 0, 6, 8, 0, 2, 6, 6, 7, 8, -1, -1, -1, -1},           /* 169 12 */
    {3, 2, 6, 3, 6, 7, 1, 0, 5, 5, 0, 4, -1, -1, -1, -1},           /* 170 10 */
    {6, 8, 2, 6, 7, 8, 2, 8, 1, 4, 5, 8, 1, 8, 5, -1},              /* 171 7 */
    {9, 4, 5, 11, 6, 1, 1, 6, 7, 1, 7, 3, -1, -1, -1, -1},          /* 172 12 */
    {1, 11, 6, 1, 6, 7, 1, 7, 0, 8, 0, 7, 9, 4, 5, -1},             /* 173 6 */
    {4, 11, 0, 4, 5, 11, 0, 11, 3, 6, 7, 11, 3, 11, 7, -1},         /* 174 7 */
    {7, 11, 6, 7, 8, 11, 5, 11, 4, 4, 11, 8, -1, -1, -1, -1},       /* 175 3 */
    {6, 5, 9, 6, 9, 10, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},      /* 176 5 */
    {3, 10, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},          /* 177 9 */
    {0, 8, 10, 0, 10, 5, 0, 5, 1, 5, 10, 6, -1, -1, -1, -1},        /* 178 14 */
    {6, 3, 10, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},       /* 179 5 */
    {1, 11, 2, 9, 10, 5, 9, 8, 10, 10, 6, 5, -1, -1, -1, -1},       /* 180 12 */
    {0, 3, 10, 0, 10, 6, 0, 6, 9, 5, 9, 6, 1, 11, 2, -1},           /* 181 6 */
    {10, 5, 8, 10, 6, 5, 8, 5, 0, 11, 2, 5, 0, 5, 2, -1},           /* 182 7 */
    {6, 3, 10, 6, 5, 3, 2, 3, 11, 11, 3, 5, -1, -1, -1, -1},        /* 183 3 */
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},           /* 184 11 */
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},        /* 185 5 */
    {1, 8, 5, 1, 0, 8, 5, 8, 6, 3, 2, 8, 6, 8, 2, -1},              /* 186 7 */
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 187 2 */
    {1, 6, 3, 1, 11, 6, 3, 6, 8, 5, 9, 6, 8, 6, 9, -1},             /* 188 7 */
    {11, 0, 1, 11, 6, 0, 9, 0, 5, 5, 0, 6, -1, -1, -1, -1},         /* 189 3 */
    {0, 8, 3, 5, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 190 4 */
    {11, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 191 1 */
    {10, 11, 5, 7, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 192 2 */
    {10, 11, 5, 10, 5, 7, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1},     /* 193 7 */
    {5, 7, 10, 5, 10, 11, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1},     /* 194 7 */
    {11, 5, 7, 11, 7, 10, 9, 1, 8, 8, 1, 3, -1, -1, -1, -1},        /* 195 10 */
    {10, 2, 1, 10, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},      /* 196 5 */
    {0, 3, 8, 1, 7, 2, 1, 5, 7, 7, 10, 2, -1, -1, -1, -1},          /* 197 12 */
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 10, -1, -1, -1, -1},          /* 198 14 */
    {7, 2, 5, 7, 10, 2, 5, 2, 9, 3, 8, 2, 9, 2, 8, -1},             /* 199 7 */
    {2, 11, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},       /* 200 5 */
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 11, 5, 2, -1, -1, -1, -1},          /* 201 11 */
    {9, 1, 0, 5, 3, 11, 5, 7, 3, 3, 2, 11, -1, -1, -1, -1},         /* 202 12 */
    {9, 2, 8, 9, 1, 2, 8, 2, 7, 11, 5, 2, 7, 2, 5, -1},             /* 203 7 */
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 204 8 */
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},        /* 205 5 */
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},        /* 206 5 */
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 207 2 */
    {5, 4, 8, 5, 8, 11, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1},     /* 208 5 */
    {5, 4, 0, 5, 0, 10, 5, 10, 11, 10, 0, 3, -1, -1, -1, -1},       /* 209 14 */
    {0, 9, 1, 8, 11, 4, 8, 10, 11, 11, 5, 4, -1, -1, -1, -1},       /* 210 12 */
    {11, 4, 10, 11, 5, 4, 10, 4, 3, 9, 1, 4, 3, 4, 1, -1},          /* 211 7 */
    {2, 1, 5, 2, 5, 8, 2, 8, 10, 4, 8, 5, -1, -1, -1, -1},          /* 212 11 */
    {0, 10, 4, 0, 3, 10, 4, 10, 5, 2, 1, 10, 5, 10, 1, -1},         /* 213 7 */
    {0, 5, 2, 0, 9, 5, 2, 5, 10, 4, 8, 5, 10, 5, 8, -1},            /* 214 7 */
    {9, 5, 4, 2, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 215 4 */
    {2, 11, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},          /* 216 9 */
    {5, 2, 11, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},       /* 217 5 */
    {3, 2, 11, 3, 11, 5, 3, 5, 8, 4, 8, 5, 0, 9, 1, -1},            /* 218 6 */
    {5, 2, 11, 5, 4, 2, 1, 2, 9, 9, 2, 4, -1, -1, -1, -1},          /* 219 3 */
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},        /* 220 5 */
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 221 2 */
    {8, 5, 4, 8, 3, 5, 9, 5, 0, 0, 5, 3, -1, -1, -1, -1},           /* 222 3 */
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 223 1 */
    {4, 7, 10, 4, 10, 9, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},    /* 224 5 */
    {0, 3, 8, 4, 7, 9, 9, 7, 10, 9, 10, 11, -1, -1, -1, -1},        /* 225 12 */
    {1, 10, 11, 1, 4, 10, 1, 0, 4, 7, 10, 4, -1, -1, -1, -1},       /* 226 11 */
    {3, 4, 1, 3, 8, 4, 1, 4, 11, 7, 10, 4, 11, 4, 10, -1},          /* 227 7 */
    {4, 7, 10, 9, 4, 10, 9, 10, 2, 9, 2, 1, -1, -1, -1, -1},        /* 228 9 */
    {9, 4, 7, 9, 7, 10, 9, 10, 1, 2, 1, 10, 0, 3, 8, -1},           /* 229 6 */
    {10, 4, 7, 10, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},      /* 230 5 */
    {10, 4, 7, 10, 2, 4, 8, 4, 3, 3, 4, 2, -1, -1, -1, -1},         /* 231 3 */
    {2, 11, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},          /* 232 14 */
    {9, 7, 11, 9, 4, 7, 11, 7, 2, 8, 0, 7, 2, 7, 0, -1},            /* 233 7 */
    {3, 11, 7, 3, 2, 11, 7, 11, 4, 1, 0, 11, 4, 11, 0, -1},         /* 234 7 */
    {1, 2, 11, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 235 4 */
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},        /* 236 5 */
    {4, 1, 9, 4, 7, 1, 0, 1, 8, 8, 1, 7, -1, -1, -1, -1},           /* 237 3 */
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 238 2 */
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 239 1 */
    {9, 8, 11, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 240 8 */
    {3, 9, 0, 3, 10, 9, 10, 11, 9, -1, -1, -1, -1, -1, -1, -1},     /* 241 5 */
    {0, 11, 1, 0, 8, 11, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},    /* 242 5 */
    {3, 11, 1, 10, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 243 2 */
    {1, 10, 2, 1, 9, 10, 9, 8, 10, -1, -1, -1, -1, -1, -1, -1},     /* 244 5 */
    {3, 9, 0, 3, 10, 9, 1, 9, 2, 2, 9, 10, -1, -1, -1, -1},         /* 245 3 */
    {0, 10, 2, 8, 10, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},   /* 246 2 */
    {3, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 247 1 */
    {2, 8, 3, 2, 11, 8, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},      /* 248 5 */
    {9, 2, 11, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    /* 249 2 */
    {2, 8, 3, 2, 11, 8, 0, 8, 1, 1, 8, 11, -1, -1, -1, -1},         /* 250 3 */
    {1, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, /* 251 1 */
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     /* 252 2 */
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 253 1 */
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  /* 254 1 */
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1}}; /* 255 0 */

__constant__ int8_t vtkMarchingCubes_edges[12][2] = {
    {0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6},
    {7, 6}, {4, 7}, {0, 4}, {1, 5}, {3, 7}, {2, 6}};

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

int main(int argc, char **argv) {
  float isoValue;
  float3 *vert;
  int3 *tri;
  int maxLevel;
  long j, nvert, ntri, numCells;
  FILE *file, *cell_file, *scalar_file;
  char xyz_path[FILENAME_MAX], tri_path[FILENAME_MAX], xdmf_path[FILENAME_MAX],
      *xyz_base, *tri_base, *cell_path, *scalar_path, *output_path;

  thrust::host_vector<Cell> h_cells;

  if (argc != 5) {
    fprintf(stderr, "iso in.cells in.scalars isoValue mesh\n");
    exit(1);
  }
  cell_path = argv[1];
  scalar_path = argv[2];
  isoValue = std::stof(argv[3]);
  output_path = argv[4];

  vec3i coordOrigin(1 << 30);
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
  for (;;) {
    Cell cell;
    if (fread(&cell, sizeof(CellCoords), 1, cell_file) != 1)
      break;
    if (fread(&cell.scalar, sizeof(cell.scalar), 1, scalar_file) != 1)
      break;
    maxLevel = std::max(maxLevel, cell.level);
    h_cells.push_back(cell);
    bounds_lower = min(bounds_lower, cell.lower);
    bounds_upper = max(bounds_upper, cell.lower + vec3i(1 << cell.level));
    coordOrigin = min(coordOrigin, cell.lower);
    numCells++;
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
  thrust::device_vector<Cell> d_cells = h_cells;
  cudaDeviceSynchronize();
  thrust::sort(d_cells.begin(), d_cells.end(),
               CompareByCoordsLowerOnly(coordOrigin));
  cudaDeviceSynchronize();
  thrust::device_vector<int> d_atomicCounter(1);
  thrust::device_vector<TriangleVertex> d_triangleVertices(0);

  {
    d_atomicCounter[0] = 0;
    size_t numJobs = 8 * h_cells.size();
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
    size_t numJobs = 8 * h_cells.size();
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
  nvert = d_vertexArray.size();
  ntri = d_indexArray.size();
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
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", output_path);
  xyz_base = xyz_path;
  tri_base = tri_path;
  for (j = 0; xyz_path[j] != '\0'; j++) {
    if (xyz_path[j] == '/' && xyz_path[j + 1] != '\0') {
      xyz_base = &xyz_path[j + 1];
      tri_base = &tri_path[j + 1];
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
          "    </Grid>\n"
          "  </Domain>\n"
          "</Xdmf>\n",
          ntri, ntri, tri_base, nvert, xyz_base);
  if (fclose(file) != 0) {
    fprintf(stderr, "iso: fail to close '%s'\n", xdmf_path);
    exit(1);
  }
}
