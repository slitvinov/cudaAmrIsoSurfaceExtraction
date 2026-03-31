#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALLOC(p, n)                                                            \
  do {                                                                         \
    (p) = malloc((n) * sizeof *(p));                                           \
    if (!(p)) {                                                                \
      fprintf(stderr, "iso2d: error: malloc failed\n");                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static const int8_t msq_cases[16][5] = {
    {-1, -1, -1, -1, -1}, {0, 3, -1, -1, -1}, {0, 1, -1, -1, -1},
    {1, 3, -1, -1, -1},   {1, 2, -1, -1, -1}, {0, 3, 1, 2, -1},
    {0, 2, -1, -1, -1},   {2, 3, -1, -1, -1}, {2, 3, -1, -1, -1},
    {0, 2, -1, -1, -1},   {0, 1, 2, 3, -1},   {1, 2, -1, -1, -1},
    {1, 3, -1, -1, -1},   {0, 1, -1, -1, -1}, {0, 3, -1, -1, -1},
    {-1, -1, -1, -1, -1},
};
static const int8_t msq_edges[4][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3}};

struct vec2i {
  int x, y;
};
struct vec2f {
  float x, y;
};
struct int2 {
  int x, y;
};
struct Vertex {
  struct vec2f position;
  float scalar, field;
  uint32_t id;
};
struct Cell {
  struct vec2i lower;
  int level;
  uint64_t morton;
  float scalar, field;
};

static struct vec2i vec2i_shr(struct vec2i v, int s) {
  struct vec2i u = {v.x >> s, v.y >> s};
  return u;
}
static int vec2i_eq(struct vec2i a, struct vec2i b) {
  return a.x == b.x && a.y == b.y;
}
static int vec2i_lt(struct vec2i a, struct vec2i b) {
  return (a.x < b.x) || (a.x == b.x && a.y < b.y);
}
static int vec2f_eq(struct vec2f a, struct vec2f b) {
  return a.x == b.x && a.y == b.y;
}
static int vec2f_lt(struct vec2f a, struct vec2f b) {
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
static long morton(int x, int y) {
  return (leftShift2((uint32_t)y) << 1) | leftShift2((uint32_t)x);
}

static struct Vertex dual(struct Cell c) {
  struct Vertex v;
  v.position.x = c.lower.x + 0.5 * (1 << c.level);
  v.position.y = c.lower.y + 0.5 * (1 << c.level);
  v.scalar = c.scalar;
  v.field = c.field;
  return v;
}

static int findActual(struct Cell *cells, unsigned long long ncell,
                      struct Cell *result, struct vec2i lower, int level) {
  int f;
  unsigned long long lo, hi, mid;
  uint64_t target;
  target = morton(lower.x, lower.y);
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
  if (vec2i_eq(vec2i_shr(result->lower, f), vec2i_shr(lower, f)))
    return 1;
  if (lo > 0) {
    *result = cells[lo - 1];
    f = level > result->level ? level : result->level;
    if (vec2i_eq(vec2i_shr(result->lower, f), vec2i_shr(lower, f)))
      return 1;
  }
  return 0;
}

static void extract(struct Cell *cells, unsigned long long ncell, float iso,
                    struct Vertex *out, unsigned long long size,
                    unsigned long long *cnt) {
  unsigned long long wid, id;
  int did, x, y, index, i, j, k, ii, dx, dy, ix, iy, skip;
  const int8_t *edge, *vert;
  float t;
  struct Vertex vertex[4], v0, v1, segVertex[2];
  struct vec2i lower;
  struct Cell corner[2][2], cell;
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
          if (!findActual(cells, ncell, &corner[iy][ix], lower, cell.level)) {
            skip = 1;
            break;
          }
          if (corner[iy][ix].level < cell.level) {
            skip = 1;
            break;
          }
          if (corner[iy][ix].level == cell.level &&
              vec2i_lt(corner[iy][ix].lower, cell.lower)) {
            skip = 1;
            break;
          }
        }
      if (skip)
        continue;
      x = dx == -1;
      y = dy == -1;
      vertex[0] = dual(corner[0 + y][0 + x]);
      vertex[1] = dual(corner[0 + y][1 - x]);
      vertex[2] = dual(corner[1 - y][1 - x]);
      vertex[3] = dual(corner[1 - y][0 + x]);
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
          t = (iso - v0.scalar) / (float)(v1.scalar - v0.scalar);
          segVertex[ii].position.x =
              (1.0 - t) * v0.position.x + t * v1.position.x;
          segVertex[ii].position.y =
              (1.0 - t) * v0.position.y + t * v1.position.y;
          segVertex[ii].scalar = (1.0 - t) * v0.scalar + t * v1.scalar;
          segVertex[ii].field = (1.0 - t) * v0.field + t * v1.field;
        }
        if (vec2f_eq(segVertex[0].position, segVertex[1].position))
          continue;
        id = (*cnt)++;
        if (out == NULL || 2 * id + 1 >= size)
          continue;
        for (j = 0; j < 2; j++) {
          k = 2 * id + j;
          out[k].position = segVertex[j].position;
          out[k].scalar = segVertex[j].scalar;
          out[k].field = segVertex[j].field;
          out[k].id = 2 * id + j;
        }
      }
    }
  }
}

static void createVertexArray(unsigned long long *cnt,
                              const struct Vertex *vertices, int nvert,
                              struct Vertex *vert, int size,
                              struct int2 *index) {
  int i, j, k, l, id, tid, *seg;
  struct Vertex vertex;
  for (tid = 0; tid < nvert; tid++) {
    vertex = vertices[tid];
    if (tid > 0 && vec2f_eq(vertex.position, vertices[tid - 1].position))
      continue;
    id = (*cnt)++;
    if (vert == NULL || id >= size)
      continue;
    vert[id].position = vertex.position;
    vert[id].scalar = vertex.scalar;
    vert[id].field = vertex.field;
    for (i = tid; i < nvert && vec2f_eq(vertices[i].position, vertex.position);
         i++) {
      j = vertices[i].id;
      k = j % 2;
      l = j / 2;
      seg = &index[l].x;
      seg[k] = id;
    }
  }
}

static int comp(const void *a, const void *b) {
  uint64_t am = ((const struct Cell *)a)->morton;
  uint64_t bm = ((const struct Cell *)b)->morton;
  return (am > bm) - (am < bm);
}
static int comp_vert(const void *a, const void *b) {
  struct vec2f ap = ((const struct Vertex *)a)->position;
  struct vec2f bp = ((const struct Vertex *)b)->position;
  if (vec2f_lt(ap, bp))
    return -1;
  if (vec2f_lt(bp, ap))
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
              "iso2d: error: bad field spec '%s' (use file:offset:stride)\n",
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
    fprintf(stderr, "iso2d: error: cannot open '%s'\n", path);
    exit(1);
  }
  sz = fsize(f);
  if (sz == (long)(ncell * stride * sizeof(float))) {
    float *buf;
    ALLOC(buf, ncell * stride);
    if (fread(buf, sizeof(float), ncell * stride, f) !=
        ncell * (unsigned long long)stride) {
      fprintf(stderr, "iso2d: error: short read '%s'\n", path);
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
      fprintf(stderr, "iso2d: error: short read '%s'\n", path);
      exit(1);
    }
    for (i = 0; i < ncell; i++)
      out[i] = (float)buf[i * stride + offset];
    free(buf);
  } else {
    fprintf(stderr,
            "iso2d: error: '%s' size %ld does not match %llu cells with stride "
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
  unsigned long long ncell, nseg, nvert, cnt, i;
  float *geo_data, *sc_data, *fl_data, h, h_min, ox, oy;
  struct Cell *cells;
  struct Vertex *tv, *vert;
  struct int2 *seg;
  char xy_path[FILENAME_MAX], seg_path[FILENAME_MAX], attr_path[FILENAME_MAX],
      xdmf_out[FILENAME_MAX];
  char *xy_base, *seg_base, *attr_base;
  long j;

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
          "Usage: iso2d [-v] coords.raw scalar field level output\n\n"
          "Extract 2D iso-lines from raw binary files.\n\n"
          "Arguments:\n"
          "  coords.raw  Quadrilateral vertices, float[ncell][4][2].\n"
          "  scalar      Cell-centered scalar (file or file:offset:stride).\n"
          "  field       Cell-centered field (file or file:offset:stride).\n"
          "  level       Iso-value.\n"
          "  output      Output file name prefix.\n");
      exit(0);
    default:
      fprintf(stderr, "iso2d: error: unknown option '%s'\n", *argv);
      exit(1);
    }
  if (!argv[0] || !argv[1] || !argv[2] || !argv[3] || !argv[4]) {
    fprintf(stderr, "Usage: iso2d [-v] coords.raw scalar.raw field.raw level "
                    "output\n");
    exit(1);
  }
  geo_path = argv[0];
  parse_field_spec(argv[1], sc_path, sizeof sc_path, &sc_off, &sc_stride);
  parse_field_spec(argv[2], fl_path, sizeof fl_path, &fl_off, &fl_stride);
  iso = strtod(argv[3], &end_p);
  if (*end_p != '\0') {
    fprintf(stderr, "iso2d: error: '%s' is not a number\n", argv[3]);
    exit(1);
  }
  output_path = argv[4];

  /* read geometry, determine ncell from file size */
  f = fopen(geo_path, "rb");
  if (!f) {
    fprintf(stderr, "iso2d: error: cannot open '%s'\n", geo_path);
    exit(1);
  }
  geo_sz = fsize(f);
  ncell = geo_sz / (4 * 2 * sizeof(float));
  if (ncell * 4 * 2 * sizeof(float) != (unsigned long long)geo_sz) {
    fprintf(stderr, "iso2d: error: '%s' size %ld is not a multiple of %lu\n",
            geo_path, geo_sz, 4 * 2 * sizeof(float));
    exit(1);
  }
  ALLOC(geo_data, 8 * ncell);
  if (fread(geo_data, sizeof(float), 8 * ncell, f) != 8 * ncell) {
    fprintf(stderr, "iso2d: error: short read '%s'\n", geo_path);
    exit(1);
  }
  fclose(f);

  /* find h_min, origin */
  h_min = 1e30f;
  ox = 1e30f;
  oy = 1e30f;
  for (i = 0; i < ncell; i++) {
    float x0 = geo_data[8 * i + 0];
    float y0 = geo_data[8 * i + 1];
    h = geo_data[8 * i + 4] - x0;
    if (h < h_min)
      h_min = h;
    if (x0 < ox)
      ox = x0;
    if (y0 < oy)
      oy = y0;
  }

  /* build cells */
  ALLOC(cells, ncell);
  for (i = 0; i < ncell; i++) {
    float x0 = geo_data[8 * i + 0];
    float y0 = geo_data[8 * i + 1];
    h = geo_data[8 * i + 4] - x0;
    cells[i].lower.x = (int)roundf((x0 - ox) / h_min);
    cells[i].lower.y = (int)roundf((y0 - oy) / h_min);
    cells[i].level = (int)roundf(log2f(h / h_min));
    cells[i].morton = morton(cells[i].lower.x, cells[i].lower.y);
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
    fprintf(stderr, "iso2d: ncell=%llu h_min=%g origin=[%g %g]\n", ncell,
            (double)h_min, (double)ox, (double)oy);

  /* sort by morton code */
  qsort(cells, ncell, sizeof *cells, comp);

  /* extract: count */
  cnt = 0;
  extract(cells, ncell, iso, NULL, 0, &cnt);
  nseg = cnt;
  if (Verbose)
    fprintf(stderr, "iso2d: nseg=%llu\n", nseg);
  if (nseg == 0) {
    fprintf(stderr, "iso2d: error: no segments\n");
    exit(1);
  }

  /* extract: fill */
  ALLOC(tv, 2 * nseg);
  cnt = 0;
  extract(cells, ncell, iso, tv, 2 * nseg, &cnt);
  free(cells);

  /* sort and deduplicate vertices */
  qsort(tv, 2 * nseg, sizeof *tv, comp_vert);
  cnt = 0;
  createVertexArray(&cnt, tv, 2 * nseg, NULL, 0, NULL);
  nvert = cnt;
  if (Verbose)
    fprintf(stderr, "iso2d: nvert=%llu\n", nvert);
  ALLOC(vert, nvert);
  ALLOC(seg, nseg);
  cnt = 0;
  createVertexArray(&cnt, tv, 2 * nseg, vert, nvert, seg);
  free(tv);

  /* write output */
  snprintf(xy_path, sizeof xy_path, "%s.xy.raw", output_path);
  snprintf(seg_path, sizeof seg_path, "%s.seg.raw", output_path);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", output_path);
  snprintf(xdmf_out, sizeof xdmf_out, "%s.xdmf2", output_path);
  xy_base = xy_path;
  seg_base = seg_path;
  attr_base = attr_path;
  for (j = 0; xy_path[j] != '\0'; j++)
    if (xy_path[j] == '/' && xy_path[j + 1] != '\0') {
      xy_base = &xy_path[j + 1];
      seg_base = &seg_path[j + 1];
      attr_base = &attr_path[j + 1];
    }

  {
    float xy[2];
    f = fopen(xy_path, "wb");
    if (!f) {
      fprintf(stderr, "iso2d: error: cannot open '%s'\n", xy_path);
      exit(1);
    }
    for (i = 0; i < nvert; i++) {
      xy[0] = vert[i].position.x * h_min + ox;
      xy[1] = vert[i].position.y * h_min + oy;
      fwrite(xy, sizeof xy, 1, f);
    }
    fclose(f);
  }

  f = fopen(seg_path, "wb");
  fwrite(seg, nseg * sizeof *seg, 1, f);
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
  free(seg);

  f = fopen(xdmf_out, "w");
  fprintf(f,
          "<Xdmf\n"
          "    Version=\"2\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Topology\n"
          "          TopologyType=\"Polyline\"\n"
          "          NodesPerElement=\"2\"\n"
          "          Dimensions=\"%llu\">\n"
          "        <DataItem\n"
          "            Dimensions=\"%llu 2\"\n"
          "            NumberType=\"Int\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Topology>\n"
          "      <Geometry\n"
          "          GeometryType=\"XY\">\n"
          "        <DataItem\n"
          "            Dimensions=\"%llu 2\"\n"
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
          nseg, nseg, seg_base, nvert, xy_base, nvert, attr_base);
  fclose(f);
}
