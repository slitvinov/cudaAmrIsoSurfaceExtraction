#include <limits.h>
#include <math.h>
#include <stdint.h>
#include "table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALLOC(p, n)                                                            \
  do {                                                                         \
    (p) = malloc((n) * sizeof *(p));                                           \
    if (!(p)) {                                                                \
      fprintf(stderr, "iso3d: error: malloc failed\n");                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct vec3i {
  int x, y, z;
};
struct vec3f {
  float x, y, z;
};
struct int3 {
  int x, y, z;
};
struct Vertex {
  struct vec3f position;
  float scalar, field;
  uint32_t id;
};
struct Cell {
  struct vec3i lower;
  int level;
  uint64_t morton;
  float scalar, field;
};

static struct vec3i vec3i_shr(struct vec3i v, int s) {
  struct vec3i u = {v.x >> s, v.y >> s, v.z >> s};
  return u;
}
static int vec3i_eq(struct vec3i a, struct vec3i b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
static int vec3i_lt(struct vec3i a, struct vec3i b) {
  return (a.x < b.x) ||
         (a.x == b.x && (a.y < b.y || (a.y == b.y && a.z < b.z)));
}
static int vec3f_eq(struct vec3f a, struct vec3f b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
static int vec3f_lt(struct vec3f a, struct vec3f b) {
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
static long morton(int x, int y, int z) {
  return (leftShift3((uint32_t)z) << 2) | (leftShift3((uint32_t)y) << 1) |
         (leftShift3((uint32_t)x) << 0);
}

static struct Vertex dual(struct Cell c) {
  struct Vertex v;
  v.position.x = c.lower.x + 0.5 * (1 << c.level);
  v.position.y = c.lower.y + 0.5 * (1 << c.level);
  v.position.z = c.lower.z + 0.5 * (1 << c.level);
  v.scalar = c.scalar;
  v.field = c.field;
  return v;
}

static int findActual(struct Cell *cells, unsigned long long ncell,
                      struct Cell *result, struct vec3i lower, int level) {
  int f;
  unsigned long long lo, hi, mid;
  uint64_t target;
  target = morton(lower.x, lower.y, lower.z);
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
  if (vec3i_eq(vec3i_shr(result->lower, f), vec3i_shr(lower, f)))
    return 1;
  if (lo > 0) {
    *result = cells[lo - 1];
    f = level > result->level ? level : result->level;
    if (vec3i_eq(vec3i_shr(result->lower, f), vec3i_shr(lower, f)))
      return 1;
  }
  return 0;
}

static void extract(struct Cell *cells, unsigned long long ncell, float iso,
                    struct Vertex *out, unsigned long long size,
                    unsigned long long *cnt) {
  unsigned long long wid, id;
  int did, x, y, z, index, i, j, k, ii, dx, dy, dz, ix, iy, iz, skip;
  const int8_t *edge, *vert;
  float t;
  struct Vertex vertex[8], v0, v1, triVertex[3];
  struct vec3i lower;
  struct Cell corner[2][2][2], cell;
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
            if (!findActual(cells, ncell, &corner[iz][iy][ix], lower,
                            cell.level)) {
              skip = 1;
              break;
            }
            if (corner[iz][iy][ix].level < cell.level) {
              skip = 1;
              break;
            }
            if (corner[iz][iy][ix].level == cell.level &&
                vec3i_lt(corner[iz][iy][ix].lower, cell.lower)) {
              skip = 1;
              break;
            }
          }
      if (skip)
        continue;
      x = dx == -1;
      y = dy == -1;
      z = dz == -1;
      vertex[0] = dual(corner[0 + z][0 + y][0 + x]);
      vertex[1] = dual(corner[0 + z][0 + y][1 - x]);
      vertex[2] = dual(corner[0 + z][1 - y][1 - x]);
      vertex[3] = dual(corner[0 + z][1 - y][0 + x]);
      vertex[4] = dual(corner[1 - z][0 + y][0 + x]);
      vertex[5] = dual(corner[1 - z][0 + y][1 - x]);
      vertex[6] = dual(corner[1 - z][1 - y][1 - x]);
      vertex[7] = dual(corner[1 - z][1 - y][0 + x]);
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
          t = (iso - v0.scalar) / (float)(v1.scalar - v0.scalar);
          triVertex[ii].position.x =
              (1.0 - t) * v0.position.x + t * v1.position.x;
          triVertex[ii].position.y =
              (1.0 - t) * v0.position.y + t * v1.position.y;
          triVertex[ii].position.z =
              (1.0 - t) * v0.position.z + t * v1.position.z;
          triVertex[ii].scalar = (1.0 - t) * v0.scalar + t * v1.scalar;
          triVertex[ii].field = (1.0 - t) * v0.field + t * v1.field;
        }
        if (vec3f_eq(triVertex[1].position, triVertex[0].position))
          continue;
        if (vec3f_eq(triVertex[2].position, triVertex[0].position))
          continue;
        if (vec3f_eq(triVertex[1].position, triVertex[2].position))
          continue;
        id = (*cnt)++;
        if (out == NULL || 3 * id + 2 >= size)
          continue;
        for (j = 0; j < 3; j++) {
          k = 3 * id + j;
          out[k].position = triVertex[j].position;
          out[k].scalar = triVertex[j].scalar;
          out[k].field = triVertex[j].field;
          out[k].id = 4 * id + j;
        }
      }
    }
  }
}

static void createVertexArray(unsigned long long *cnt,
                              const struct Vertex *vertices, int nvert,
                              struct Vertex *vert, int size,
                              struct int3 *index) {
  int i, j, k, l, id, tid, *tri;
  struct Vertex vertex;
  for (tid = 0; tid < nvert; tid++) {
    vertex = vertices[tid];
    if (tid > 0 && vec3f_eq(vertex.position, vertices[tid - 1].position))
      continue;
    id = (*cnt)++;
    if (vert == NULL || id >= size)
      continue;
    vert[id].position = vertex.position;
    vert[id].scalar = vertex.scalar;
    vert[id].field = vertex.field;
    for (i = tid; i < nvert && vec3f_eq(vertices[i].position, vertex.position);
         i++) {
      j = vertices[i].id;
      k = j % 4;
      l = j / 4;
      tri = &index[l].x;
      tri[k] = id;
    }
  }
}

static int comp(const void *a, const void *b) {
  uint64_t am = ((const struct Cell *)a)->morton;
  uint64_t bm = ((const struct Cell *)b)->morton;
  return (am > bm) - (am < bm);
}
static int comp_vert(const void *a, const void *b) {
  struct vec3f ap = ((const struct Vertex *)a)->position;
  struct vec3f bp = ((const struct Vertex *)b)->position;
  if (vec3f_lt(ap, bp))
    return -1;
  if (vec3f_lt(bp, ap))
    return 1;
  return 0;
}

static long fsize(FILE *f) {
  long n;
  fseek(f, 0, SEEK_END);
  n = ftell(f);
  fseek(f, 0, SEEK_SET);
  return n;
}

static void parse_field_spec(const char *spec, char *path, int pathsz,
                             int *offset, int *stride) {
  const char *p;
  int n;
  *offset = 0;
  *stride = 1;
  p = strchr(spec, ':');
  if (p) {
    n = (int)(p - spec);
    if (n >= pathsz)
      n = pathsz - 1;
    memcpy(path, spec, n);
    path[n] = '\0';
    if (sscanf(p + 1, "%d:%d", offset, stride) != 2) {
      fprintf(stderr,
              "iso3d: error: bad field spec '%s' (use file:offset:stride)\n",
              spec);
      exit(1);
    }
  } else {
    n = strlen(spec);
    if (n >= pathsz)
      n = pathsz - 1;
    memcpy(path, spec, n);
    path[n] = '\0';
  }
}

static void read_field(const char *path, unsigned long long ncell, int offset,
                       int stride, float *out) {
  unsigned long long i;
  long sz;
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "iso3d: error: cannot open '%s'\n", path);
    exit(1);
  }
  sz = fsize(f);
  if (sz == (long)(ncell * stride * sizeof(float))) {
    float *buf;
    ALLOC(buf, ncell * stride);
    if (fread(buf, sizeof(float), ncell * stride, f) !=
        ncell * (unsigned long long)stride) {
      fprintf(stderr, "iso3d: error: short read '%s'\n", path);
      exit(1);
    }
    for (i = 0; i < ncell; i++)
      out[i] = buf[i * stride + offset];
    free(buf);
  } else if (sz == (long)(ncell * stride * sizeof(double))) {
    double *buf;
    ALLOC(buf, ncell * stride);
    if (fread(buf, sizeof(double), ncell * stride, f) !=
        ncell * (unsigned long long)stride) {
      fprintf(stderr, "iso3d: error: short read '%s'\n", path);
      exit(1);
    }
    for (i = 0; i < ncell; i++)
      out[i] = (float)buf[i * stride + offset];
    free(buf);
  } else {
    fprintf(stderr,
            "iso3d: error: '%s' size %ld does not match %llu cells with stride "
            "%d\n",
            path, sz, ncell, stride);
    exit(1);
  }
  fclose(f);
}

int main(int argc, char **argv) {
  int Verbose;
  char *geo_path, *output_path, *end_p;
  char sc_path[FILENAME_MAX], fl_path[FILENAME_MAX];
  int sc_off, sc_stride, fl_off, fl_stride;
  float iso;
  FILE *f;
  long geo_sz;
  unsigned long long ncell, ntri, nvert, cnt, i;
  float *geo_data, *sc_data, *fl_data, h, h_min, ox, oy, oz;
  struct Cell *cells;
  struct Vertex *tv, *vert;
  struct int3 *tri;
  char xyz_path[FILENAME_MAX], tri_path[FILENAME_MAX], attr_path[FILENAME_MAX],
      xdmf_out[FILENAME_MAX];
  char *xyz_base, *tri_base, *attr_base;
  long j;
  int kk;

  (void)argc;
  Verbose = 0;
  while (*++argv != NULL && argv[0][0] == '-')
    switch (argv[0][1]) {
    case 'v':
      Verbose = 1;
      break;
    case 'h':
      fprintf(
          stderr,
          "Usage: iso3d [-v] coords.raw scalar field level output\n\n"
          "Extract 3D iso-surfaces from raw binary files.\n\n"
          "Arguments:\n"
          "  coords.raw  Hexahedron vertices, float[ncell][8][3].\n"
          "  scalar      Cell-centered scalar (file or file:offset:stride).\n"
          "  field       Cell-centered field (file or file:offset:stride).\n"
          "  level       Iso-value.\n"
          "  output      Output file name prefix.\n");
      exit(0);
    default:
      fprintf(stderr, "iso3d: error: unknown option '%s'\n", *argv);
      exit(1);
    }
  if (!argv[0] || !argv[1] || !argv[2] || !argv[3] || !argv[4]) {
    fprintf(stderr, "Usage: iso3d [-v] coords.raw scalar.raw field.raw level "
                    "output\n");
    exit(1);
  }
  geo_path = argv[0];
  parse_field_spec(argv[1], sc_path, sizeof sc_path, &sc_off, &sc_stride);
  parse_field_spec(argv[2], fl_path, sizeof fl_path, &fl_off, &fl_stride);
  iso = strtod(argv[3], &end_p);
  if (*end_p != '\0') {
    fprintf(stderr, "iso3d: error: '%s' is not a number\n", argv[3]);
    exit(1);
  }
  output_path = argv[4];

  /* read geometry, determine ncell from file size */
  f = fopen(geo_path, "rb");
  if (!f) {
    fprintf(stderr, "iso3d: error: cannot open '%s'\n", geo_path);
    exit(1);
  }
  geo_sz = fsize(f);
  ncell = geo_sz / (8 * 3 * sizeof(float));
  if (ncell * 8 * 3 * sizeof(float) != (unsigned long long)geo_sz) {
    fprintf(stderr, "iso3d: error: '%s' size %ld is not a multiple of %lu\n",
            geo_path, geo_sz, 8 * 3 * sizeof(float));
    exit(1);
  }
  ALLOC(geo_data, 24 * ncell);
  if (fread(geo_data, sizeof(float), 24 * ncell, f) != 24 * ncell) {
    fprintf(stderr, "iso3d: error: short read '%s'\n", geo_path);
    exit(1);
  }
  fclose(f);

  /* find h_min, origin from hex vertices */
  h_min = 1e30f;
  ox = 1e30f;
  oy = 1e30f;
  oz = 1e30f;
  for (i = 0; i < ncell; i++) {
    float *v = &geo_data[24 * i];
    float xmin = v[0], xmax = v[0], ymin = v[1], zmin = v[2];
    for (kk = 1; kk < 8; kk++) {
      if (v[3 * kk] < xmin)
        xmin = v[3 * kk];
      if (v[3 * kk] > xmax)
        xmax = v[3 * kk];
      if (v[3 * kk + 1] < ymin)
        ymin = v[3 * kk + 1];
      if (v[3 * kk + 2] < zmin)
        zmin = v[3 * kk + 2];
    }
    h = xmax - xmin;
    if (h < h_min)
      h_min = h;
    if (xmin < ox)
      ox = xmin;
    if (ymin < oy)
      oy = ymin;
    if (zmin < oz)
      oz = zmin;
  }

  /* build cells */
  ALLOC(cells, ncell);
  for (i = 0; i < ncell; i++) {
    float *v = &geo_data[24 * i];
    float xmin = v[0], xmax = v[0], ymin = v[1], zmin = v[2];
    for (kk = 1; kk < 8; kk++) {
      if (v[3 * kk] < xmin)
        xmin = v[3 * kk];
      if (v[3 * kk] > xmax)
        xmax = v[3 * kk];
      if (v[3 * kk + 1] < ymin)
        ymin = v[3 * kk + 1];
      if (v[3 * kk + 2] < zmin)
        zmin = v[3 * kk + 2];
    }
    h = xmax - xmin;
    cells[i].lower.x = (int)roundf((xmin - ox) / h_min);
    cells[i].lower.y = (int)roundf((ymin - oy) / h_min);
    cells[i].lower.z = (int)roundf((zmin - oz) / h_min);
    cells[i].level = (int)roundf(log2f(h / h_min));
    cells[i].morton =
        morton(cells[i].lower.x, cells[i].lower.y, cells[i].lower.z);
  }
  free(geo_data);

  /* read fields */
  ALLOC(sc_data, ncell);
  ALLOC(fl_data, ncell);
  read_field(sc_path, ncell, sc_off, sc_stride, sc_data);
  read_field(fl_path, ncell, fl_off, fl_stride, fl_data);
  for (i = 0; i < ncell; i++) {
    cells[i].scalar = sc_data[i];
    cells[i].field = fl_data[i];
  }
  free(sc_data);
  free(fl_data);

  if (Verbose)
    fprintf(stderr, "iso3d: ncell=%llu h_min=%g origin=[%g %g %g]\n", ncell,
            (double)h_min, (double)ox, (double)oy, (double)oz);

  /* sort by morton code */
  qsort(cells, ncell, sizeof *cells, comp);

  /* extract: count */
  cnt = 0;
  extract(cells, ncell, iso, NULL, 0, &cnt);
  ntri = cnt;
  if (Verbose)
    fprintf(stderr, "iso3d: ntri=%llu\n", ntri);
  if (ntri == 0) {
    fprintf(stderr, "iso3d: error: no triangles\n");
    exit(1);
  }

  /* extract: fill */
  ALLOC(tv, 3 * ntri);
  cnt = 0;
  extract(cells, ncell, iso, tv, 3 * ntri, &cnt);
  free(cells);

  /* sort and deduplicate vertices */
  qsort(tv, 3 * ntri, sizeof *tv, comp_vert);
  cnt = 0;
  createVertexArray(&cnt, tv, 3 * ntri, NULL, 0, NULL);
  nvert = cnt;
  if (Verbose)
    fprintf(stderr, "iso3d: nvert=%llu\n", nvert);
  ALLOC(vert, nvert);
  ALLOC(tri, ntri);
  cnt = 0;
  createVertexArray(&cnt, tv, 3 * ntri, vert, nvert, tri);
  free(tv);

  /* write output */
  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", output_path);
  snprintf(tri_path, sizeof tri_path, "%s.tri.raw", output_path);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", output_path);
  snprintf(xdmf_out, sizeof xdmf_out, "%s.xdmf2", output_path);
  xyz_base = xyz_path;
  tri_base = tri_path;
  attr_base = attr_path;
  for (j = 0; xyz_path[j] != '\0'; j++)
    if (xyz_path[j] == '/' && xyz_path[j + 1] != '\0') {
      xyz_base = &xyz_path[j + 1];
      tri_base = &tri_path[j + 1];
      attr_base = &attr_path[j + 1];
    }

  {
    float xyz[3];
    f = fopen(xyz_path, "wb");
    if (!f) {
      fprintf(stderr, "iso3d: error: cannot open '%s'\n", xyz_path);
      exit(1);
    }
    for (i = 0; i < nvert; i++) {
      xyz[0] = vert[i].position.x * h_min + ox;
      xyz[1] = vert[i].position.y * h_min + oy;
      xyz[2] = vert[i].position.z * h_min + oz;
      fwrite(xyz, sizeof xyz, 1, f);
    }
    fclose(f);
  }

  f = fopen(tri_path, "wb");
  fwrite(tri, ntri * sizeof *tri, 1, f);
  fclose(f);

  {
    float *a;
    ALLOC(a, nvert);
    for (i = 0; i < nvert; i++)
      a[i] = vert[i].field;
    f = fopen(attr_path, "wb");
    fwrite(a, nvert * sizeof *a, 1, f);
    fclose(f);
    free(a);
  }
  free(vert);
  free(tri);

  f = fopen(xdmf_out, "w");
  fprintf(f,
          "<Xdmf\n"
          "    Version=\"2\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Topology\n"
          "          TopologyType=\"Triangle\"\n"
          "          Dimensions=\"%llu\">\n"
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
  fclose(f);
}
