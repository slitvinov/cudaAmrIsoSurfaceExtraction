#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
   Marching cubes table (3D)
   ================================================================ */
#define __constant__ static const
#include "table.inc"

/* ================================================================
   Marching squares table (2D)
   ================================================================ */
static const int8_t msq_cases[16][5] = {
    {-1, -1, -1, -1, -1}, {0, 3, -1, -1, -1}, {0, 1, -1, -1, -1},
    {1, 3, -1, -1, -1},   {1, 2, -1, -1, -1}, {0, 3, 1, 2, -1},
    {0, 2, -1, -1, -1},   {2, 3, -1, -1, -1}, {2, 3, -1, -1, -1},
    {0, 2, -1, -1, -1},   {0, 1, 2, 3, -1},   {1, 2, -1, -1, -1},
    {1, 3, -1, -1, -1},   {0, 1, -1, -1, -1}, {0, 3, -1, -1, -1},
    {-1, -1, -1, -1, -1},
};
static const int8_t msq_edges[4][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3}};

/* ================================================================
   2D types and functions
   ================================================================ */
struct v2i {
  int x, y;
};
struct v2f {
  float x, y;
};
struct i2 {
  int x, y;
};
struct Vert2 {
  struct v2f pos;
  float scalar, field;
  uint32_t id;
};
struct Cell2 {
  struct v2i lower;
  int level;
  uint64_t morton;
  float scalar, field;
};

static struct v2i v2i_shr(struct v2i v, int s) {
  struct v2i u = {v.x >> s, v.y >> s};
  return u;
}
static int v2i_eq(struct v2i a, struct v2i b) {
  return a.x == b.x && a.y == b.y;
}
static int v2i_lt(struct v2i a, struct v2i b) {
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}
static int v2f_eq(struct v2f a, struct v2f b) {
  return a.x == b.x && a.y == b.y;
}
static int v2f_lt(struct v2f a, struct v2f b) {
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}

static long leftShift2(long x) {
  x = (x | x << 16) & 0x0000FFFF0000FFFFull;
  x = (x | x << 8) & 0x00FF00FF00FF00FFull;
  x = (x | x << 4) & 0x0F0F0F0F0F0F0F0Full;
  x = (x | x << 2) & 0x3333333333333333ull;
  x = (x | x << 1) & 0x5555555555555555ull;
  return x;
}
static long morton2(int x, int y) {
  return (leftShift2((uint32_t)y) << 1) | leftShift2((uint32_t)x);
}

static struct Vert2 dual2(struct Cell2 c) {
  struct Vert2 v;
  v.pos.x = c.lower.x + 0.5f * (1 << c.level);
  v.pos.y = c.lower.y + 0.5f * (1 << c.level);
  v.scalar = c.scalar;
  v.field = c.field;
  return v;
}

static int findActual2(struct Cell2 *cells, unsigned long long ncell,
                       struct Cell2 *result, struct v2i lower, int level) {
  int f;
  unsigned long long lo, hi, mid;
  uint64_t target = morton2(lower.x, lower.y);
  lo = 0;
  hi = ncell;
  while (lo < hi) {
    mid = lo + (hi - lo) / 2;
    if (cells[mid].morton < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == ncell)
    return 0;
  *result = cells[lo];
  f = level > result->level ? level : result->level;
  if (v2i_eq(v2i_shr(result->lower, f), v2i_shr(lower, f)))
    return 1;
  if (lo > 0) {
    *result = cells[lo - 1];
    f = level > result->level ? level : result->level;
    if (v2i_eq(v2i_shr(result->lower, f), v2i_shr(lower, f)))
      return 1;
  }
  return 0;
}

static void extract2d(struct Cell2 *cells, unsigned long long ncell, float iso,
                      struct Vert2 *out, unsigned long long size,
                      unsigned long long *cnt) {
  unsigned long long wid, id;
  int did, x, y, index, i, j, k, ii, dx, dy, ix, iy, skip;
  const int8_t *edge, *vert;
  float t;
  struct Vert2 vertex[4], v0, v1, sv[2];
  struct v2i lower;
  struct Cell2 corner[2][2], cell;
  for (wid = 0; wid < ncell; wid++) {
    for (did = 0; did < 4; did++) {
      cell = cells[wid];
      dy = (did & 2) ? 1 : -1;
      dx = (did & 1) ? 1 : -1;
      skip = 0;
      for (iy = 0; iy < 2 && !skip; iy++)
        for (ix = 0; ix < 2 && !skip; ix++) {
          lower.x = cell.lower.x + dx * ix * (1 << cell.level);
          lower.y = cell.lower.y + dy * iy * (1 << cell.level);
          if (!findActual2(cells, ncell, &corner[iy][ix], lower, cell.level)) {
            skip = 1;
            break;
          }
          if (corner[iy][ix].level < cell.level) {
            skip = 1;
            break;
          }
          if (corner[iy][ix].level == cell.level &&
              v2i_lt(corner[iy][ix].lower, cell.lower)) {
            skip = 1;
            break;
          }
        }
      if (skip)
        continue;
      x = dx == -1;
      y = dy == -1;
      vertex[0] = dual2(corner[0 + y][0 + x]);
      vertex[1] = dual2(corner[0 + y][1 - x]);
      vertex[2] = dual2(corner[1 - y][1 - x]);
      vertex[3] = dual2(corner[1 - y][0 + x]);
      index = 0;
      for (i = 0; i < 4; i++)
        if (vertex[i].scalar > iso)
          index += (1 << i);
      if (index == 0 || index == 0xf)
        continue;
      for (edge = &msq_cases[index][0]; edge[0] > -1; edge += 2) {
        for (ii = 0; ii < 2; ii++) {
          vert = msq_edges[edge[ii]];
          v0 = vertex[vert[0]];
          v1 = vertex[vert[1]];
          t = (iso - v0.scalar) / (v1.scalar - v0.scalar);
          sv[ii].pos.x = (1.0f - t) * v0.pos.x + t * v1.pos.x;
          sv[ii].pos.y = (1.0f - t) * v0.pos.y + t * v1.pos.y;
          sv[ii].scalar = (1.0f - t) * v0.scalar + t * v1.scalar;
          sv[ii].field = (1.0f - t) * v0.field + t * v1.field;
        }
        if (v2f_eq(sv[0].pos, sv[1].pos))
          continue;
        id = (*cnt)++;
        if (out == NULL || 2 * id + 1 >= size)
          continue;
        for (j = 0; j < 2; j++) {
          k = 2 * id + j;
          out[k].pos = sv[j].pos;
          out[k].scalar = sv[j].scalar;
          out[k].field = sv[j].field;
          out[k].id = 2 * id + j;
        }
      }
    }
  }
}

static void createVA2(unsigned long long *cnt, const struct Vert2 *verts,
                      int nv, struct Vert2 *out, int size, struct i2 *idx) {
  int i, j, k, l, id, tid, *seg;
  struct Vert2 v;
  for (tid = 0; tid < nv; tid++) {
    v = verts[tid];
    if (tid > 0 && v2f_eq(v.pos, verts[tid - 1].pos))
      continue;
    id = (*cnt)++;
    if (out == NULL || id >= size)
      continue;
    out[id] = v;
    for (i = tid; i < nv && v2f_eq(verts[i].pos, v.pos); i++) {
      j = verts[i].id;
      k = j % 2;
      l = j / 2;
      seg = &idx[l].x;
      seg[k] = id;
    }
  }
}

static int comp2(const void *a, const void *b) {
  uint64_t am = ((const struct Cell2 *)a)->morton;
  uint64_t bm = ((const struct Cell2 *)b)->morton;
  return (am > bm) - (am < bm);
}
static int compv2(const void *a, const void *b) {
  struct v2f ap = ((const struct Vert2 *)a)->pos;
  struct v2f bp = ((const struct Vert2 *)b)->pos;
  if (v2f_lt(ap, bp))
    return -1;
  if (v2f_lt(bp, ap))
    return 1;
  return 0;
}

/* ================================================================
   3D types and functions
   ================================================================ */
struct v3i {
  int x, y, z;
};
struct v3f {
  float x, y, z;
};
struct i3 {
  int x, y, z;
};
struct Vert3 {
  struct v3f pos;
  float scalar, field;
  uint32_t id;
};
struct Cell3 {
  struct v3i lower;
  int level;
  uint64_t morton;
  float scalar, field;
};

static struct v3i v3i_shr(struct v3i v, int s) {
  struct v3i u = {v.x >> s, v.y >> s, v.z >> s};
  return u;
}
static int v3i_eq(struct v3i a, struct v3i b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
static int v3i_lt(struct v3i a, struct v3i b) {
  return (a.x < b.x) ||
         (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}
static int v3f_eq(struct v3f a, struct v3f b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
static int v3f_lt(struct v3f a, struct v3f b) {
  return (a.x < b.x) ||
         (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}

static long leftShift3(long x) {
  x = (x | x << 32) & 0x1f00000000ffffull;
  x = (x | x << 16) & 0x1f0000ff0000ffull;
  x = (x | x << 8) & 0x100f00f00f00f00full;
  x = (x | x << 4) & 0x10c30c30c30c30c3ull;
  x = (x | x << 2) & 0x1249249249249249ull;
  return x;
}
static long morton3(int x, int y, int z) {
  return (leftShift3((uint32_t)z) << 2) | (leftShift3((uint32_t)y) << 1) |
         (leftShift3((uint32_t)x) << 0);
}

static struct Vert3 dual3(struct Cell3 c) {
  struct Vert3 v;
  v.pos.x = c.lower.x + 0.5f * (1 << c.level);
  v.pos.y = c.lower.y + 0.5f * (1 << c.level);
  v.pos.z = c.lower.z + 0.5f * (1 << c.level);
  v.scalar = c.scalar;
  v.field = c.field;
  return v;
}

static int findActual3(struct Cell3 *cells, unsigned long long ncell,
                       struct Cell3 *result, struct v3i lower, int level) {
  int f;
  unsigned long long lo, hi, mid;
  uint64_t target = morton3(lower.x, lower.y, lower.z);
  lo = 0;
  hi = ncell;
  while (lo < hi) {
    mid = lo + (hi - lo) / 2;
    if (cells[mid].morton < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == ncell)
    return 0;
  *result = cells[lo];
  f = level > result->level ? level : result->level;
  if (v3i_eq(v3i_shr(result->lower, f), v3i_shr(lower, f)))
    return 1;
  if (lo > 0) {
    *result = cells[lo - 1];
    f = level > result->level ? level : result->level;
    if (v3i_eq(v3i_shr(result->lower, f), v3i_shr(lower, f)))
      return 1;
  }
  return 0;
}

static void extract3d(struct Cell3 *cells, unsigned long long ncell, float iso,
                      struct Vert3 *out, unsigned long long size,
                      unsigned long long *cnt) {
  unsigned long long wid, id;
  int did, x, y, z, index, i, j, k, ii, dx, dy, dz, ix, iy, iz, skip;
  const int8_t *edge, *vert;
  float t;
  struct Vert3 vertex[8], v0, v1, tv[3];
  struct v3i lower;
  struct Cell3 corner[2][2][2], cell;
  for (wid = 0; wid < ncell; wid++) {
    for (did = 0; did < 8; did++) {
      cell = cells[wid];
      dz = (did & 4) ? 1 : -1;
      dy = (did & 2) ? 1 : -1;
      dx = (did & 1) ? 1 : -1;
      skip = 0;
      for (iz = 0; iz < 2 && !skip; iz++)
        for (iy = 0; iy < 2 && !skip; iy++)
          for (ix = 0; ix < 2 && !skip; ix++) {
            lower.x = cell.lower.x + dx * ix * (1 << cell.level);
            lower.y = cell.lower.y + dy * iy * (1 << cell.level);
            lower.z = cell.lower.z + dz * iz * (1 << cell.level);
            if (!findActual3(cells, ncell, &corner[iz][iy][ix], lower,
                             cell.level)) {
              skip = 1;
              break;
            }
            if (corner[iz][iy][ix].level < cell.level) {
              skip = 1;
              break;
            }
            if (corner[iz][iy][ix].level == cell.level &&
                v3i_lt(corner[iz][iy][ix].lower, cell.lower)) {
              skip = 1;
              break;
            }
          }
      if (skip)
        continue;
      x = dx == -1;
      y = dy == -1;
      z = dz == -1;
      vertex[0] = dual3(corner[0 + z][0 + y][0 + x]);
      vertex[1] = dual3(corner[0 + z][0 + y][1 - x]);
      vertex[2] = dual3(corner[0 + z][1 - y][1 - x]);
      vertex[3] = dual3(corner[0 + z][1 - y][0 + x]);
      vertex[4] = dual3(corner[1 - z][0 + y][0 + x]);
      vertex[5] = dual3(corner[1 - z][0 + y][1 - x]);
      vertex[6] = dual3(corner[1 - z][1 - y][1 - x]);
      vertex[7] = dual3(corner[1 - z][1 - y][0 + x]);
      index = 0;
      for (i = 0; i < 8; i++)
        if (vertex[i].scalar > iso)
          index += (1 << i);
      if (index == 0 || index == 0xff)
        continue;
      for (edge = &vtkMarchingCubesTriangleCases[index][0]; edge[0] > -1;
           edge += 3) {
        for (ii = 0; ii < 3; ii++) {
          vert = vtkMarchingCubes_edges[edge[ii]];
          v0 = vertex[vert[0]];
          v1 = vertex[vert[1]];
          t = (iso - v0.scalar) / (v1.scalar - v0.scalar);
          tv[ii].pos.x = (1.0f - t) * v0.pos.x + t * v1.pos.x;
          tv[ii].pos.y = (1.0f - t) * v0.pos.y + t * v1.pos.y;
          tv[ii].pos.z = (1.0f - t) * v0.pos.z + t * v1.pos.z;
          tv[ii].scalar = (1.0f - t) * v0.scalar + t * v1.scalar;
          tv[ii].field = (1.0f - t) * v0.field + t * v1.field;
        }
        if (v3f_eq(tv[1].pos, tv[0].pos))
          continue;
        if (v3f_eq(tv[2].pos, tv[0].pos))
          continue;
        if (v3f_eq(tv[1].pos, tv[2].pos))
          continue;
        id = (*cnt)++;
        if (out == NULL || 3 * id + 2 >= size)
          continue;
        for (j = 0; j < 3; j++) {
          k = 3 * id + j;
          out[k].pos = tv[j].pos;
          out[k].scalar = tv[j].scalar;
          out[k].field = tv[j].field;
          out[k].id = 4 * id + j;
        }
      }
    }
  }
}

static void createVA3(unsigned long long *cnt, const struct Vert3 *verts,
                      int nv, struct Vert3 *out, int size, struct i3 *idx) {
  int i, j, k, l, id, tid, *tri;
  struct Vert3 v;
  for (tid = 0; tid < nv; tid++) {
    v = verts[tid];
    if (tid > 0 && v3f_eq(v.pos, verts[tid - 1].pos))
      continue;
    id = (*cnt)++;
    if (out == NULL || id >= size)
      continue;
    out[id] = v;
    for (i = tid; i < nv && v3f_eq(verts[i].pos, v.pos); i++) {
      j = verts[i].id;
      k = j % 4;
      l = j / 4;
      tri = &idx[l].x;
      tri[k] = id;
    }
  }
}

static int comp3(const void *a, const void *b) {
  uint64_t am = ((const struct Cell3 *)a)->morton;
  uint64_t bm = ((const struct Cell3 *)b)->morton;
  return (am > bm) - (am < bm);
}
static int compv3(const void *a, const void *b) {
  struct v3f ap = ((const struct Vert3 *)a)->pos;
  struct v3f bp = ((const struct Vert3 *)b)->pos;
  if (v3f_lt(ap, bp))
    return -1;
  if (v3f_lt(bp, ap))
    return 1;
  return 0;
}

/* ================================================================
   Python: extract2d
   ================================================================ */
enum { MAX_SEG = 4, MAX_TRI = 5 };

static PyObject *py_extract2d(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  static char *kw[] = {"coords", "scalar", "field", "iso", "out", "work",
                       NULL};
  PyObject *co, *so, *fo, *out_obj = Py_None, *work_obj = Py_None;
  double iso;
  unsigned long long nc, ns, nv, cnt, i;
  float *geo, *sc, *fl, h, hm, ox, oy;
  struct Cell2 *cells;
  struct Vert2 *tv, *vert;
  npy_intp dims[2];
  int uw, uo;
  size_t wn;
  PyArrayObject *ca, *sa, *fa;

  (void)self;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOd|OO", kw, &co, &so, &fo,
                                   &iso, &out_obj, &work_obj))
    return NULL;
  ca = (PyArrayObject *)PyArray_FROM_OTF(co, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  sa = (PyArrayObject *)PyArray_FROM_OTF(so, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  fa = (PyArrayObject *)PyArray_FROM_OTF(fo, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  if (!ca || !sa || !fa) {
    Py_XDECREF(ca);
    Py_XDECREF(sa);
    Py_XDECREF(fa);
    return NULL;
  }
  nc = PyArray_SIZE(sa);
  if (PyArray_SIZE(fa) != (npy_intp)nc ||
      PyArray_SIZE(ca) != (npy_intp)(nc * 8)) {
    PyErr_SetString(PyExc_ValueError, "array size mismatch");
    Py_DECREF(ca);
    Py_DECREF(sa);
    Py_DECREF(fa);
    return NULL;
  }
  uw = (work_obj != Py_None);
  uo = (out_obj != Py_None);
  wn = nc * sizeof(struct Cell2) + 2 * MAX_SEG * nc * sizeof(struct Vert2);
  if (uw) {
    if (!PyArray_Check(work_obj) ||
        (size_t)PyArray_NBYTES((PyArrayObject *)work_obj) < wn) {
      PyErr_Format(PyExc_ValueError, "work too small: need %zu bytes", wn);
      Py_DECREF(ca);
      Py_DECREF(sa);
      Py_DECREF(fa);
      return NULL;
    }
  }
  geo = (float *)PyArray_DATA(ca);
  sc = (float *)PyArray_DATA(sa);
  fl = (float *)PyArray_DATA(fa);
  hm = 1e30f;
  ox = 1e30f;
  oy = 1e30f;
  for (i = 0; i < nc; i++) {
    float x0 = geo[8 * i], y0 = geo[8 * i + 1];
    h = geo[8 * i + 4] - x0;
    if (h < hm)
      hm = h;
    if (x0 < ox)
      ox = x0;
    if (y0 < oy)
      oy = y0;
  }
  if (uw) {
    char *wp = (char *)PyArray_DATA((PyArrayObject *)work_obj);
    cells = (struct Cell2 *)wp;
    tv = (struct Vert2 *)(wp + nc * sizeof(struct Cell2));
  } else {
    cells = malloc(nc * sizeof *cells);
    tv = NULL;
    if (!cells) {
      Py_DECREF(ca);
      Py_DECREF(sa);
      Py_DECREF(fa);
      return PyErr_NoMemory();
    }
  }
  for (i = 0; i < nc; i++) {
    float x0 = geo[8 * i], y0 = geo[8 * i + 1];
    h = geo[8 * i + 4] - x0;
    cells[i].lower.x = (int)roundf((x0 - ox) / hm);
    cells[i].lower.y = (int)roundf((y0 - oy) / hm);
    cells[i].level = (int)roundf(log2f(h / hm));
    cells[i].morton = morton2(cells[i].lower.x, cells[i].lower.y);
    cells[i].scalar = sc[i];
    cells[i].field = fl[i];
  }
  Py_DECREF(ca);
  Py_DECREF(sa);
  Py_DECREF(fa);
  qsort(cells, nc, sizeof *cells, comp2);
  cnt = 0;
  extract2d(cells, nc, (float)iso, NULL, 0, &cnt);
  ns = cnt;
  if (ns == 0) {
    if (!uw)
      free(cells);
    if (uo)
      return Py_BuildValue("KK", (unsigned long long)0, (unsigned long long)0);
    dims[0] = 0;
    dims[1] = 2;
    return Py_BuildValue("NNN", PyArray_ZEROS(2, dims, NPY_FLOAT, 0),
                         PyArray_ZEROS(2, dims, NPY_INT32, 0),
                         PyArray_ZEROS(1, dims, NPY_FLOAT, 0));
  }
  if (!uw) {
    tv = malloc(2 * ns * sizeof *tv);
    if (!tv) {
      free(cells);
      return PyErr_NoMemory();
    }
  }
  cnt = 0;
  extract2d(cells, nc, (float)iso, tv, 2 * ns, &cnt);
  qsort(tv, 2 * ns, sizeof *tv, compv2);
  cnt = 0;
  createVA2(&cnt, tv, 2 * ns, NULL, 0, NULL);
  nv = cnt;
  if (uw)
    vert = (void *)cells;
  else {
    vert = malloc(nv * sizeof *vert);
    if (!vert) {
      free(tv);
      return PyErr_NoMemory();
    }
  }
  {
    PyObject *xo = NULL, *so2 = NULL, *ao = NULL;
    float *xd, *ad;
    struct i2 *sb;
    if (uo && PyTuple_Check(out_obj) && PyTuple_Size(out_obj) == 3) {
      PyArrayObject *o0 = (PyArrayObject *)PyTuple_GetItem(out_obj, 0);
      PyArrayObject *o1 = (PyArrayObject *)PyTuple_GetItem(out_obj, 1);
      PyArrayObject *o2 = (PyArrayObject *)PyTuple_GetItem(out_obj, 2);
      if (PyArray_SIZE(o0) >= (npy_intp)(nv * 2) &&
          PyArray_SIZE(o1) >= (npy_intp)(ns * 2) &&
          PyArray_SIZE(o2) >= (npy_intp)nv) {
        xo = (PyObject *)o0;
        so2 = (PyObject *)o1;
        ao = (PyObject *)o2;
      } else {
        if (!uw) {
          free(vert);
          free(tv);
        }
        PyErr_SetString(PyExc_ValueError, "out arrays too small");
        return NULL;
      }
    } else if (!uo) {
      dims[0] = nv;
      dims[1] = 2;
      xo = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      dims[0] = ns;
      so2 = PyArray_SimpleNew(2, dims, NPY_INT32);
      dims[0] = nv;
      ao = PyArray_SimpleNew(1, dims, NPY_FLOAT);
      if (!xo || !so2 || !ao) {
        if (!uw) {
          free(vert);
          free(tv);
        }
        Py_XDECREF(xo);
        Py_XDECREF(so2);
        Py_XDECREF(ao);
        return NULL;
      }
    }
    sb = (struct i2 *)PyArray_DATA((PyArrayObject *)so2);
    cnt = 0;
    createVA2(&cnt, tv, 2 * ns, vert, nv, sb);
    if (!uw)
      free(tv);
    xd = (float *)PyArray_DATA((PyArrayObject *)xo);
    ad = (float *)PyArray_DATA((PyArrayObject *)ao);
    for (i = 0; i < nv; i++) {
      xd[2 * i] = vert[i].pos.x * hm + ox;
      xd[2 * i + 1] = vert[i].pos.y * hm + oy;
      ad[i] = vert[i].field;
    }
    if (!uw)
      free(vert);
    if (uo)
      return Py_BuildValue("KK", ns, nv);
    return Py_BuildValue("NNN", xo, so2, ao);
  }
}

/* ================================================================
   Python: extract3d
   ================================================================ */
static PyObject *py_extract3d(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  static char *kw[] = {"coords", "scalar", "field", "iso", "out", "work",
                       NULL};
  PyObject *co, *so, *fo, *out_obj = Py_None, *work_obj = Py_None;
  double iso;
  unsigned long long nc, nt, nv, cnt, i;
  float *geo, *sc, *fl, h, hm, ox, oy, oz;
  struct Cell3 *cells;
  struct Vert3 *tv, *vert;
  npy_intp dims[2];
  int uw, uo, kk;
  size_t wn;
  PyArrayObject *ca, *sa, *fa;

  (void)self;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOd|OO", kw, &co, &so, &fo,
                                   &iso, &out_obj, &work_obj))
    return NULL;
  ca = (PyArrayObject *)PyArray_FROM_OTF(co, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  sa = (PyArrayObject *)PyArray_FROM_OTF(so, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  fa = (PyArrayObject *)PyArray_FROM_OTF(fo, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  if (!ca || !sa || !fa) {
    Py_XDECREF(ca);
    Py_XDECREF(sa);
    Py_XDECREF(fa);
    return NULL;
  }
  nc = PyArray_SIZE(sa);
  if (PyArray_SIZE(fa) != (npy_intp)nc ||
      PyArray_SIZE(ca) != (npy_intp)(nc * 24)) {
    PyErr_SetString(PyExc_ValueError, "array size mismatch");
    Py_DECREF(ca);
    Py_DECREF(sa);
    Py_DECREF(fa);
    return NULL;
  }
  uw = (work_obj != Py_None);
  uo = (out_obj != Py_None);
  wn = nc * sizeof(struct Cell3) + 3 * MAX_TRI * nc * sizeof(struct Vert3);
  if (uw) {
    if (!PyArray_Check(work_obj) ||
        (size_t)PyArray_NBYTES((PyArrayObject *)work_obj) < wn) {
      PyErr_Format(PyExc_ValueError, "work too small: need %zu bytes", wn);
      Py_DECREF(ca);
      Py_DECREF(sa);
      Py_DECREF(fa);
      return NULL;
    }
  }
  geo = (float *)PyArray_DATA(ca);
  sc = (float *)PyArray_DATA(sa);
  fl = (float *)PyArray_DATA(fa);
  hm = 1e30f;
  ox = 1e30f;
  oy = 1e30f;
  oz = 1e30f;
  for (i = 0; i < nc; i++) {
    float *v = &geo[24 * i];
    float xn = v[0], xx = v[0], yn = v[1], zn = v[2];
    for (kk = 1; kk < 8; kk++) {
      if (v[3 * kk] < xn)
        xn = v[3 * kk];
      if (v[3 * kk] > xx)
        xx = v[3 * kk];
      if (v[3 * kk + 1] < yn)
        yn = v[3 * kk + 1];
      if (v[3 * kk + 2] < zn)
        zn = v[3 * kk + 2];
    }
    h = xx - xn;
    if (h < hm)
      hm = h;
    if (xn < ox)
      ox = xn;
    if (yn < oy)
      oy = yn;
    if (zn < oz)
      oz = zn;
  }
  if (uw) {
    char *wp = (char *)PyArray_DATA((PyArrayObject *)work_obj);
    cells = (struct Cell3 *)wp;
    tv = (struct Vert3 *)(wp + nc * sizeof(struct Cell3));
  } else {
    cells = malloc(nc * sizeof *cells);
    tv = NULL;
    if (!cells) {
      Py_DECREF(ca);
      Py_DECREF(sa);
      Py_DECREF(fa);
      return PyErr_NoMemory();
    }
  }
  for (i = 0; i < nc; i++) {
    float *v = &geo[24 * i];
    float xn = v[0], xx = v[0], yn = v[1], zn = v[2];
    for (kk = 1; kk < 8; kk++) {
      if (v[3 * kk] < xn)
        xn = v[3 * kk];
      if (v[3 * kk] > xx)
        xx = v[3 * kk];
      if (v[3 * kk + 1] < yn)
        yn = v[3 * kk + 1];
      if (v[3 * kk + 2] < zn)
        zn = v[3 * kk + 2];
    }
    h = xx - xn;
    cells[i].lower.x = (int)roundf((xn - ox) / hm);
    cells[i].lower.y = (int)roundf((yn - oy) / hm);
    cells[i].lower.z = (int)roundf((zn - oz) / hm);
    cells[i].level = (int)roundf(log2f(h / hm));
    cells[i].morton = morton3(cells[i].lower.x, cells[i].lower.y,
                              cells[i].lower.z);
    cells[i].scalar = sc[i];
    cells[i].field = fl[i];
  }
  Py_DECREF(ca);
  Py_DECREF(sa);
  Py_DECREF(fa);
  qsort(cells, nc, sizeof *cells, comp3);
  cnt = 0;
  extract3d(cells, nc, (float)iso, NULL, 0, &cnt);
  nt = cnt;
  if (nt == 0) {
    if (!uw)
      free(cells);
    if (uo)
      return Py_BuildValue("KK", (unsigned long long)0, (unsigned long long)0);
    dims[0] = 0;
    dims[1] = 3;
    return Py_BuildValue("NNN", PyArray_ZEROS(2, dims, NPY_FLOAT, 0),
                         PyArray_ZEROS(2, dims, NPY_INT32, 0),
                         PyArray_ZEROS(1, dims, NPY_FLOAT, 0));
  }
  if (!uw) {
    tv = malloc(3 * nt * sizeof *tv);
    if (!tv) {
      free(cells);
      return PyErr_NoMemory();
    }
  }
  cnt = 0;
  extract3d(cells, nc, (float)iso, tv, 3 * nt, &cnt);
  qsort(tv, 3 * nt, sizeof *tv, compv3);
  cnt = 0;
  createVA3(&cnt, tv, 3 * nt, NULL, 0, NULL);
  nv = cnt;
  if (uw)
    vert = (void *)cells;
  else {
    vert = malloc(nv * sizeof *vert);
    if (!vert) {
      free(tv);
      return PyErr_NoMemory();
    }
  }
  {
    PyObject *xo = NULL, *to = NULL, *ao = NULL;
    float *xd, *ad;
    struct i3 *tb;
    if (uo && PyTuple_Check(out_obj) && PyTuple_Size(out_obj) == 3) {
      PyArrayObject *o0 = (PyArrayObject *)PyTuple_GetItem(out_obj, 0);
      PyArrayObject *o1 = (PyArrayObject *)PyTuple_GetItem(out_obj, 1);
      PyArrayObject *o2 = (PyArrayObject *)PyTuple_GetItem(out_obj, 2);
      if (PyArray_SIZE(o0) >= (npy_intp)(nv * 3) &&
          PyArray_SIZE(o1) >= (npy_intp)(nt * 3) &&
          PyArray_SIZE(o2) >= (npy_intp)nv) {
        xo = (PyObject *)o0;
        to = (PyObject *)o1;
        ao = (PyObject *)o2;
      } else {
        if (!uw) {
          free(vert);
          free(tv);
        }
        PyErr_SetString(PyExc_ValueError, "out arrays too small");
        return NULL;
      }
    } else if (!uo) {
      dims[0] = nv;
      dims[1] = 3;
      xo = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      dims[0] = nt;
      to = PyArray_SimpleNew(2, dims, NPY_INT32);
      dims[0] = nv;
      ao = PyArray_SimpleNew(1, dims, NPY_FLOAT);
      if (!xo || !to || !ao) {
        if (!uw) {
          free(vert);
          free(tv);
        }
        Py_XDECREF(xo);
        Py_XDECREF(to);
        Py_XDECREF(ao);
        return NULL;
      }
    }
    tb = (struct i3 *)PyArray_DATA((PyArrayObject *)to);
    cnt = 0;
    createVA3(&cnt, tv, 3 * nt, vert, nv, tb);
    if (!uw)
      free(tv);
    xd = (float *)PyArray_DATA((PyArrayObject *)xo);
    ad = (float *)PyArray_DATA((PyArrayObject *)ao);
    for (i = 0; i < nv; i++) {
      xd[3 * i] = vert[i].pos.x * hm + ox;
      xd[3 * i + 1] = vert[i].pos.y * hm + oy;
      xd[3 * i + 2] = vert[i].pos.z * hm + oz;
      ad[i] = vert[i].field;
    }
    if (!uw)
      free(vert);
    if (uo)
      return Py_BuildValue("KK", nt, nv);
    return Py_BuildValue("NNN", xo, to, ao);
  }
}

/* ================================================================
   Python: workspace_size2d / workspace_size3d / example2d / example3d
   ================================================================ */
static PyObject *py_ws2(PyObject *self, PyObject *args) {
  unsigned long long nc;
  (void)self;
  if (!PyArg_ParseTuple(args, "K", &nc))
    return NULL;
  return PyLong_FromSize_t(nc * sizeof(struct Cell2) +
                           2 * MAX_SEG * nc * sizeof(struct Vert2));
}

static PyObject *py_ws3(PyObject *self, PyObject *args) {
  unsigned long long nc;
  (void)self;
  if (!PyArg_ParseTuple(args, "K", &nc))
    return NULL;
  return PyLong_FromSize_t(nc * sizeof(struct Cell3) +
                           3 * MAX_TRI * nc * sizeof(struct Vert3));
}

static PyObject *py_ex2(PyObject *self, PyObject *args) {
  int nx = 20, ny = 20, ix, iy, k;
  npy_intp cd[3], sd[1];
  PyObject *co, *so;
  float *c, *s, cx, cy, rx, ry;
  (void)self;
  if (!PyArg_ParseTuple(args, "|ii", &nx, &ny))
    return NULL;
  cd[0] = nx * ny;
  cd[1] = 4;
  cd[2] = 2;
  sd[0] = nx * ny;
  co = PyArray_SimpleNew(3, cd, NPY_FLOAT);
  so = PyArray_SimpleNew(1, sd, NPY_FLOAT);
  if (!co || !so) {
    Py_XDECREF(co);
    Py_XDECREF(so);
    return NULL;
  }
  c = (float *)PyArray_DATA((PyArrayObject *)co);
  s = (float *)PyArray_DATA((PyArrayObject *)so);
  cx = nx / 2.0f;
  cy = ny / 2.0f;
  rx = nx / 4.0f;
  ry = ny / 5.0f;
  k = 0;
  for (ix = 0; ix < nx; ix++)
    for (iy = 0; iy < ny; iy++) {
      float x0 = (float)ix, y0 = (float)iy;
      c[k++] = x0;
      c[k++] = y0;
      c[k++] = x0;
      c[k++] = y0 + 1;
      c[k++] = x0 + 1;
      c[k++] = y0 + 1;
      c[k++] = x0 + 1;
      c[k++] = y0;
      s[ix * ny + iy] = (x0 - cx) * (x0 - cx) / (rx * rx) +
                         (y0 - cy) * (y0 - cy) / (ry * ry) - 1;
    }
  return Py_BuildValue("NN", co, so);
}

static PyObject *py_ex3(PyObject *self, PyObject *args) {
  int nx = 10, ny = 10, nz = 10, ix, iy, iz, n, k;
  npy_intp cd[3], sd[1];
  PyObject *co, *so;
  float *c, *s, cx, cy, cz, rx, ry, rz;
  (void)self;
  if (!PyArg_ParseTuple(args, "|iii", &nx, &ny, &nz))
    return NULL;
  n = nx * ny * nz;
  cd[0] = n;
  cd[1] = 8;
  cd[2] = 3;
  sd[0] = n;
  co = PyArray_SimpleNew(3, cd, NPY_FLOAT);
  so = PyArray_SimpleNew(1, sd, NPY_FLOAT);
  if (!co || !so) {
    Py_XDECREF(co);
    Py_XDECREF(so);
    return NULL;
  }
  c = (float *)PyArray_DATA((PyArrayObject *)co);
  s = (float *)PyArray_DATA((PyArrayObject *)so);
  cx = nx / 2.0f;
  cy = ny / 2.0f;
  cz = nz / 2.0f;
  rx = nx / 4.0f;
  ry = ny / 5.0f;
  rz = nz / 6.0f;
  k = 0;
  n = 0;
  for (ix = 0; ix < nx; ix++)
    for (iy = 0; iy < ny; iy++)
      for (iz = 0; iz < nz; iz++) {
        float x0 = (float)ix, y0 = (float)iy, z0 = (float)iz;
        int dx, dy, dz;
        for (dx = 0; dx < 2; dx++)
          for (dy = 0; dy < 2; dy++)
            for (dz = 0; dz < 2; dz++) {
              c[k++] = x0 + dx;
              c[k++] = y0 + dy;
              c[k++] = z0 + dz;
            }
        s[n++] = (x0 - cx) * (x0 - cx) / (rx * rx) +
                 (y0 - cy) * (y0 - cy) / (ry * ry) +
                 (z0 - cz) * (z0 - cz) / (rz * rz) - 1;
      }
  return Py_BuildValue("NN", co, so);
}

/* ================================================================
   Python: dump2d / dump3d
   ================================================================ */
static PyObject *py_dump2d(PyObject *self, PyObject *args) {
  const char *prefix;
  PyArrayObject *xy_a, *seg_a, *attr_a;
  PyObject *xy_o, *seg_o, *attr_o;
  npy_intp nvert, nseg;
  float *xy, *attr;
  int *seg;
  char path[FILENAME_MAX];
  char *base;
  long j;
  FILE *f;

  (void)self;
  if (!PyArg_ParseTuple(args, "sOOO", &prefix, &xy_o, &seg_o, &attr_o))
    return NULL;
  xy_a = (PyArrayObject *)PyArray_FROM_OTF(xy_o, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  seg_a =
      (PyArrayObject *)PyArray_FROM_OTF(seg_o, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  attr_a =
      (PyArrayObject *)PyArray_FROM_OTF(attr_o, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  if (!xy_a || !seg_a || !attr_a) {
    Py_XDECREF(xy_a);
    Py_XDECREF(seg_a);
    Py_XDECREF(attr_a);
    return NULL;
  }
  nvert = PyArray_SIZE(attr_a);
  nseg = PyArray_SIZE(seg_a) / 2;
  xy = (float *)PyArray_DATA(xy_a);
  seg = (int *)PyArray_DATA(seg_a);
  attr = (float *)PyArray_DATA(attr_a);

  snprintf(path, sizeof path, "%s.xy.raw", prefix);
  f = fopen(path, "wb");
  if (!f) {
    PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
    goto fail2;
  }
  fwrite(xy, sizeof(float), 2 * nvert, f);
  fclose(f);

  snprintf(path, sizeof path, "%s.seg.raw", prefix);
  f = fopen(path, "wb");
  if (!f) {
    PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
    goto fail2;
  }
  fwrite(seg, sizeof(int), 2 * nseg, f);
  fclose(f);

  snprintf(path, sizeof path, "%s.attr.raw", prefix);
  f = fopen(path, "wb");
  if (!f) {
    PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
    goto fail2;
  }
  fwrite(attr, sizeof(float), nvert, f);
  fclose(f);

  base = path;
  snprintf(path, sizeof path, "%s.xy.raw", prefix);
  for (j = 0; path[j]; j++)
    if (path[j] == '/' && path[j + 1])
      base = &path[j + 1];

  {
    char xy_base[FILENAME_MAX], seg_base[FILENAME_MAX], attr_base[FILENAME_MAX];
    snprintf(xy_base, sizeof xy_base, "%.*s.xy.raw",
             (int)(base - path + strlen(base) - 7), base);
    snprintf(seg_base, sizeof seg_base, "%.*s.seg.raw",
             (int)(base - path + strlen(base) - 7), base);
    snprintf(attr_base, sizeof attr_base, "%.*s.attr.raw",
             (int)(base - path + strlen(base) - 7), base);

    snprintf(path, sizeof path, "%s.xdmf2", prefix);
    f = fopen(path, "w");
    if (!f) {
      PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
      goto fail2;
    }

    base = path;
    for (j = 0; path[j]; j++)
      if (path[j] == '/' && path[j + 1])
        base = &path[j + 1];

    {
      int off = (int)(base - path);
      const char *p = prefix + (strlen(prefix) - (strlen(path) - off - 6));
      snprintf(xy_base, sizeof xy_base, "%s.xy.raw", p);
      snprintf(seg_base, sizeof seg_base, "%s.seg.raw", p);
      snprintf(attr_base, sizeof attr_base, "%s.attr.raw", p);
    }

    fprintf(f,
            "<Xdmf\n"
            "    Version=\"2\">\n"
            "  <Domain>\n"
            "    <Grid>\n"
            "      <Topology\n"
            "          TopologyType=\"Polyline\"\n"
            "          NodesPerElement=\"2\"\n"
            "          Dimensions=\"%ld\">\n"
            "        <DataItem\n"
            "            Dimensions=\"%ld 2\"\n"
            "            NumberType=\"Int\"\n"
            "            Format=\"Binary\">\n"
            "          %s\n"
            "        </DataItem>\n"
            "      </Topology>\n"
            "      <Geometry\n"
            "          GeometryType=\"XY\">\n"
            "        <DataItem\n"
            "            Dimensions=\"%ld 2\"\n"
            "            Precision=\"4\"\n"
            "            Format=\"Binary\">\n"
            "          %s\n"
            "        </DataItem>\n"
            "      </Geometry>\n"
            "      <Attribute\n"
            "          Center=\"Node\"\n"
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
            nseg, nseg, seg_base, nvert, xy_base, nvert, attr_base);
    fclose(f);
  }

  Py_DECREF(xy_a);
  Py_DECREF(seg_a);
  Py_DECREF(attr_a);
  Py_RETURN_NONE;
fail2:
  Py_DECREF(xy_a);
  Py_DECREF(seg_a);
  Py_DECREF(attr_a);
  return NULL;
}

static PyObject *py_dump3d(PyObject *self, PyObject *args) {
  const char *prefix;
  PyArrayObject *xyz_a, *tri_a, *attr_a;
  PyObject *xyz_o, *tri_o, *attr_o;
  npy_intp nvert, ntri;
  float *xyz, *attr;
  int *tri;
  char path[FILENAME_MAX], xyz_base[FILENAME_MAX], tri_base[FILENAME_MAX],
      attr_base[FILENAME_MAX];
  char *base;
  long j;
  FILE *f;

  (void)self;
  if (!PyArg_ParseTuple(args, "sOOO", &prefix, &xyz_o, &tri_o, &attr_o))
    return NULL;
  xyz_a =
      (PyArrayObject *)PyArray_FROM_OTF(xyz_o, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  tri_a =
      (PyArrayObject *)PyArray_FROM_OTF(tri_o, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  attr_a =
      (PyArrayObject *)PyArray_FROM_OTF(attr_o, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
  if (!xyz_a || !tri_a || !attr_a) {
    Py_XDECREF(xyz_a);
    Py_XDECREF(tri_a);
    Py_XDECREF(attr_a);
    return NULL;
  }
  nvert = PyArray_SIZE(attr_a);
  ntri = PyArray_SIZE(tri_a) / 3;
  xyz = (float *)PyArray_DATA(xyz_a);
  tri = (int *)PyArray_DATA(tri_a);
  attr = (float *)PyArray_DATA(attr_a);

  snprintf(path, sizeof path, "%s.xyz.raw", prefix);
  f = fopen(path, "wb");
  if (!f) {
    PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
    goto fail3;
  }
  fwrite(xyz, sizeof(float), 3 * nvert, f);
  fclose(f);

  snprintf(path, sizeof path, "%s.tri.raw", prefix);
  f = fopen(path, "wb");
  if (!f) {
    PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
    goto fail3;
  }
  fwrite(tri, sizeof(int), 3 * ntri, f);
  fclose(f);

  snprintf(path, sizeof path, "%s.attr.raw", prefix);
  f = fopen(path, "wb");
  if (!f) {
    PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
    goto fail3;
  }
  fwrite(attr, sizeof(float), nvert, f);
  fclose(f);

  snprintf(path, sizeof path, "%s.xdmf2", prefix);
  base = path;
  for (j = 0; path[j]; j++)
    if (path[j] == '/' && path[j + 1])
      base = &path[j + 1];
  {
    int off = (int)(base - path);
    snprintf(xyz_base, sizeof xyz_base, "%s.xyz.raw", prefix + off);
    snprintf(tri_base, sizeof tri_base, "%s.tri.raw", prefix + off);
    snprintf(attr_base, sizeof attr_base, "%s.attr.raw", prefix + off);
  }

  f = fopen(path, "w");
  if (!f) {
    PyErr_Format(PyExc_IOError, "cannot open '%s'", path);
    goto fail3;
  }
  fprintf(f,
          "<Xdmf\n"
          "    Version=\"2\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Topology\n"
          "          TopologyType=\"Triangle\"\n"
          "          Dimensions=\"%ld\">\n"
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
          "          Center=\"Node\"\n"
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
  fclose(f);

  Py_DECREF(xyz_a);
  Py_DECREF(tri_a);
  Py_DECREF(attr_a);
  Py_RETURN_NONE;
fail3:
  Py_DECREF(xyz_a);
  Py_DECREF(tri_a);
  Py_DECREF(attr_a);
  return NULL;
}

/* ================================================================
   Module definition
   ================================================================ */
static PyMethodDef methods[] = {
    {"extract2d", (PyCFunction)py_extract2d, METH_VARARGS | METH_KEYWORDS,
     "extract2d(coords, scalar, field, iso, out=None, work=None)\n\n"
     "Extract 2D iso-lines from AMR quadrilateral mesh.\n\n"
     "Parameters\n"
     "----------\n"
     "coords : float32 array (ncell, 4, 2) or flat (ncell*8,)\n"
     "    Quad vertices: (x0,y0), (x0,y1), (x1,y1), (x1,y0) per cell.\n"
     "scalar : float32 array (ncell,)\n"
     "    Cell-centered scalar field for iso-line extraction.\n"
     "field : float32 array (ncell,)\n"
     "    Cell-centered field to interpolate onto vertices.\n"
     "iso : float\n"
     "    Iso-value.\n"
     "out : tuple (xy, seg, attr), optional\n"
     "    Pre-allocated output arrays.\n"
     "work : uint8 array, optional\n"
     "    Workspace from workspace_size2d(ncell).\n\n"
     "Returns\n"
     "-------\n"
     "Without out: (xy(nvert,2), seg(nseg,2), attr(nvert,))\n"
     "With out: (nseg, nvert)"},
    {"extract3d", (PyCFunction)py_extract3d, METH_VARARGS | METH_KEYWORDS,
     "extract3d(coords, scalar, field, iso, out=None, work=None)\n\n"
     "Extract 3D iso-surfaces from AMR hexahedral mesh.\n\n"
     "Parameters\n"
     "----------\n"
     "coords : float32 array (ncell, 8, 3) or flat (ncell*24,)\n"
     "    Hexahedron vertices per cell.\n"
     "scalar : float32 array (ncell,)\n"
     "    Cell-centered scalar field for iso-surface extraction.\n"
     "field : float32 array (ncell,)\n"
     "    Cell-centered field to interpolate onto vertices.\n"
     "iso : float\n"
     "    Iso-value.\n"
     "out : tuple (xyz, tri, attr), optional\n"
     "    Pre-allocated output arrays.\n"
     "work : uint8 array, optional\n"
     "    Workspace from workspace_size3d(ncell).\n\n"
     "Returns\n"
     "-------\n"
     "Without out: (xyz(nvert,3), tri(ntri,3), attr(nvert,))\n"
     "With out: (ntri, nvert)"},
    {"workspace_size2d", py_ws2, METH_VARARGS,
     "workspace_size2d(ncell) -> int\n\n"
     "Workspace size in bytes for 2D extraction with ncell cells."},
    {"workspace_size3d", py_ws3, METH_VARARGS,
     "workspace_size3d(ncell) -> int\n\n"
     "Workspace size in bytes for 3D extraction with ncell cells."},
    {"example2d", py_ex2, METH_VARARGS,
     "example2d(nx=20, ny=20) -> (coords, scalar)\n\n"
     "Create example 2D quad mesh with ellipse scalar field."},
    {"example3d", py_ex3, METH_VARARGS,
     "example3d(nx=10, ny=10, nz=10) -> (coords, scalar)\n\n"
     "Create example 3D hex mesh with ellipsoid scalar field."},
    {"dump2d", py_dump2d, METH_VARARGS,
     "dump2d(prefix, xy, seg, attr)\n\n"
     "Write 2D iso-line to XDMF2 + binary files.\n\n"
     "Writes prefix.xy.raw, prefix.seg.raw, prefix.attr.raw,\n"
     "and prefix.xdmf2 (Polyline topology, XY geometry).\n\n"
     "Parameters\n"
     "----------\n"
     "prefix : str\n"
     "    Output path prefix (e.g., 'results/iso').\n"
     "xy : float32 array (nvert, 2)\n"
     "    Vertex positions.\n"
     "seg : int32 array (nseg, 2)\n"
     "    Segment connectivity.\n"
     "attr : float32 array (nvert,)\n"
     "    Vertex attribute.\n\n"
     "Example\n"
     "-------\n"
     "::\n\n"
     "    xy, seg, attr = amriso.extract2d(coords, scalar, field, 0.5)\n"
     "    amriso.dump2d('iso', xy, seg, attr)\n"},
    {"dump3d", py_dump3d, METH_VARARGS,
     "dump3d(prefix, xyz, tri, attr)\n\n"
     "Write 3D iso-surface to XDMF2 + binary files.\n\n"
     "Writes prefix.xyz.raw, prefix.tri.raw, prefix.attr.raw,\n"
     "and prefix.xdmf2 (Triangle topology, XYZ geometry).\n\n"
     "Parameters\n"
     "----------\n"
     "prefix : str\n"
     "    Output path prefix (e.g., 'results/iso').\n"
     "xyz : float32 array (nvert, 3)\n"
     "    Vertex positions.\n"
     "tri : int32 array (ntri, 3)\n"
     "    Triangle connectivity.\n"
     "attr : float32 array (nvert,)\n"
     "    Vertex attribute.\n\n"
     "Example\n"
     "-------\n"
     "::\n\n"
     "    xyz, tri, attr = amriso.extract3d(coords, scalar, field, 0.5)\n"
     "    amriso.dump3d('iso', xyz, tri, attr)\n"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "amriso",
    "AMR iso-surface and iso-line extraction.\n\n"
    "Extract iso-lines (2D) and iso-surfaces (3D) from cell-centered\n"
    "scalar fields on adaptive mesh refinement (AMR) grids using\n"
    "marching squares/cubes on the dual mesh.\n\n"
    "Functions\n"
    "---------\n"
    "extract2d  Extract 2D iso-lines (segments).\n"
    "extract3d  Extract 3D iso-surfaces (triangles).\n"
    "dump2d     Write 2D result to XDMF2 + binary.\n"
    "dump3d     Write 3D result to XDMF2 + binary.\n"
    "workspace_size2d  Workspace size for 2D.\n"
    "workspace_size3d  Workspace size for 3D.\n"
    "example2d  Create example 2D data.\n"
    "example3d  Create example 3D data.\n\n"
    "Quick start\n"
    "-----------\n"
    "::\n\n"
    "    import amriso\n"
    "    coords, scalar = amriso.example2d()\n"
    "    xy, seg, attr = amriso.extract2d(coords, scalar, scalar, 0.0)\n\n"
    "    coords, scalar = amriso.example3d()\n"
    "    xyz, tri, attr = amriso.extract3d(coords, scalar, scalar, 0.0)\n\n"
    "Reference: Wald, arXiv:2004.08475 (2020).\n",
    -1, methods};

PyMODINIT_FUNC PyInit_amriso(void) {
  import_array();
  return PyModule_Create(&module);
}
