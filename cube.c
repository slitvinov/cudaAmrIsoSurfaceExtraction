#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const int8_t marchingSquaresCases[16][5] = {
    {-1, -1, -1, -1, -1},
    {0, 3, -1, -1, -1},
    {0, 1, -1, -1, -1},
    {1, 3, -1, -1, -1},
    {1, 2, -1, -1, -1},
    {0, 3, 1, 2, -1},
    {0, 2, -1, -1, -1},
    {2, 3, -1, -1, -1},
    {2, 3, -1, -1, -1},
    {0, 2, -1, -1, -1},
    {0, 1, 2, 3, -1},
    {1, 2, -1, -1, -1},
    {1, 3, -1, -1, -1},
    {0, 1, -1, -1, -1},
    {0, 3, -1, -1, -1},
    {-1, -1, -1, -1, -1},
};

static const int8_t marchingSquares_edges[4][2] = {
    {0, 1},
    {1, 2},
    {3, 2},
    {0, 3},
};

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
  struct vec2i u;
  u.x = v.x >> s;
  u.y = v.y >> s;
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
                    struct Vertex *out, int size, unsigned long long *cnt) {
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
      for (edge = &marchingSquaresCases[index][0]; edge[0] > -1; edge += 2) {
        for (ii = 0; ii < 2; ii++) {
          vert = marchingSquares_edges[edge[ii]];
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
        if (out == NULL || id >= (unsigned long long)(2 * size))
          continue;
        for (j = 0; j < 2; j++) {
          k = 2 * id + j;
          out[k].position.x = segVertex[j].position.x;
          out[k].position.y = segVertex[j].position.y;
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
    vert[id].position.x = vertex.position.x;
    vert[id].position.y = vertex.position.y;
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

static int comp(const void *av, const void *bv) {
  const struct Cell *a, *b;
  a = (const struct Cell *)av;
  b = (const struct Cell *)bv;
  if (a->morton < b->morton)
    return -1;
  if (a->morton > b->morton)
    return 1;
  return 0;
}

static int comp_vert(const void *av, const void *bv) {
  const struct Vertex *a, *b;
  a = (const struct Vertex *)av;
  b = (const struct Vertex *)bv;
  if (vec2f_lt(a->position, b->position))
    return -1;
  if (vec2f_lt(b->position, a->position))
    return 1;
  return 0;
}

int main(int argc, char **argv) {
  double X0, Y0, L;
  float iso, *attr, xy[2];
  struct int2 *seg;
  int Verbose, maxlevel;
  long j, size;
  FILE *file, *cell_file, *scalar_file, *field_file;
  int cell[3], ox, oy, minlevel, ScaleFlag;
  char attr_path[FILENAME_MAX], xy_path[FILENAME_MAX], seg_path[FILENAME_MAX],
      xdmf_path[FILENAME_MAX], *attr_base, *xy_base, *seg_base, *cell_path,
      *scalar_path, *field_path, *output_path, *end;
  struct Cell *cells;
  struct Vertex *tv, *vert;
  unsigned long long nvert, nseg, ncell, cnt, i;

  (void)argc;
  Verbose = 0;
  ScaleFlag = 0;
  while (*++argv != NULL && argv[0][0] == '-')
    switch (argv[0][1]) {
    case 'h':
      fprintf(
          stderr,
          "Usage: cube [-v] [-s X0 Y0 L minlevel] in.cells in.scalar "
          "in.field iso mesh\n\n"
          "Example:\n"
          "  cube -v data.cells data.scalar data.field 0.5 output\n\n"
          "Arguments:\n"
          "  in.cells   Binary file describing the 2D AMR cell structure.\n"
          "  in.scalar  Binary file with scalar field values.\n"
          "  in.field   Binary file with additional field values.\n"
          "  iso        Iso-line value to extract (e.g., 0.5).\n"
          "  mesh       Output file name prefix for generated mesh.\n\n"
          "Options:\n"
          "  -s         Domain center, size, and minimum level for rescaling\n"
          "  -v         Enable verbose output.\n"
          "  -h         Show this help message and exit.\n");
      exit(1);
    case 's':
      argv++;
      if (argv[0] == NULL || argv[1] == NULL || argv[2] == NULL ||
          argv[3] == NULL) {
        fprintf(stderr, "cube: error: -s needs four arguments\n");
        exit(1);
      }
      ScaleFlag = 1;
      X0 = strtod(*argv, &end);
      if (*end != '\0') {
        fprintf(stderr, "cube: error: '%s' is not a double\n", *argv);
        exit(1);
      }
      argv++;
      Y0 = strtod(*argv, &end);
      if (*end != '\0') {
        fprintf(stderr, "cube: error: '%s' is not a double\n", *argv);
        exit(1);
      }
      argv++;
      L = strtod(*argv, &end);
      if (*end != '\0') {
        fprintf(stderr, "cube: error: '%s' is not a double\n", *argv);
        exit(1);
      }
      argv++;
      minlevel = strtol(*argv, &end, 10);
      if (*end != '\0') {
        fprintf(stderr, "cube: error: '%s' is not an integer\n", *argv);
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
      fprintf(stderr, "cube: error: unknown option '%s'\n", *argv);
      exit(1);
    }
positional:
  if ((cell_path = *argv++) == NULL) {
    fprintf(stderr, "cube: error: in.cells is not given\n");
    exit(1);
  }
  if ((scalar_path = *argv++) == NULL) {
    fprintf(stderr, "cube: error: in.scalar is not given\n");
    exit(1);
  }
  if ((field_path = *argv++) == NULL) {
    fprintf(stderr, "cube: error: in.field is not given\n");
    exit(1);
  }
  if (*argv == NULL) {
    fprintf(stderr, "cube: error: iso is not given\n");
    exit(1);
  }
  iso = strtod(*argv, &end);
  if (*end != '\0') {
    fprintf(stderr, "cube: error: '%s' is not a number\n", *argv);
    exit(1);
  }
  argv++;
  if ((output_path = *argv++) == NULL) {
    fprintf(stderr, "cube: error: out.mesh is not given\n");
    exit(1);
  }
  if ((cell_file = fopen(cell_path, "r")) == NULL) {
    fprintf(stderr, "cube: error: fail to open '%s'\n", cell_path);
    exit(1);
  }
  if ((scalar_file = fopen(scalar_path, "r")) == NULL) {
    fprintf(stderr, "cube: error: fail to open '%s'\n", scalar_path);
    exit(1);
  }
  if ((field_file = fopen(field_path, "r")) == NULL) {
    fprintf(stderr, "cube: error: fail to open '%s'\n", field_path);
    exit(1);
  }
  fseek(cell_file, 0, SEEK_END);
  size = ftell(cell_file);
  fseek(cell_file, 0, SEEK_SET);
  ncell = size / (3 * sizeof(int));
  if ((cells = (struct Cell *)malloc(ncell * sizeof *cells)) == NULL) {
    fprintf(stderr, "cube: error: malloc failed\n");
    exit(1);
  }
  ox = INT_MAX;
  oy = INT_MAX;
  maxlevel = 0;
  for (i = 0; i < ncell; i++) {
    if (fread(cell, sizeof(cell), 1, cell_file) != 1) {
      fprintf(stderr, "cube: error: fail to read '%s'\n", cell_path);
      exit(1);
    }
    cells[i].lower.x = cell[0];
    cells[i].lower.y = cell[1];
    cells[i].level = cell[2];
    if (fread(&cells[i].scalar, sizeof(cells[i].scalar), 1, scalar_file) != 1) {
      fprintf(stderr, "cube: error: fail to read '%s'\n", scalar_path);
      exit(1);
    }
    if (fread(&cells[i].field, sizeof(cells[i].field), 1, field_file) != 1) {
      fprintf(stderr, "cube: error: fail to read '%s'\n", field_path);
      exit(1);
    }
    if (cells[i].level > maxlevel)
      maxlevel = cells[i].level;
    if (cells[i].lower.x < ox)
      ox = cells[i].lower.x;
    if (cells[i].lower.y < oy)
      oy = cells[i].lower.y;
  }
  for (i = 0; i < ncell; i++) {
    cells[i].lower.x -= ox;
    cells[i].lower.y -= oy;
    cells[i].morton = morton(cells[i].lower.x, cells[i].lower.y);
  }

  if (Verbose)
    fprintf(stderr, "cube: ncell, maxlevel, origin: %llu %d [%d %d]\n", ncell,
            maxlevel, ox, oy);
  if (fclose(cell_file) != 0) {
    fprintf(stderr, "cube: error: fail to close '%s'\n", cell_path);
    exit(1);
  }
  if (fclose(scalar_file) != 0) {
    fprintf(stderr, "cube: error: fail to close '%s'\n", scalar_path);
    exit(1);
  }
  if (fclose(field_file) != 0) {
    fprintf(stderr, "cube: error: fail to close '%s'\n", field_path);
    exit(1);
  }
  qsort(cells, ncell, sizeof *cells, comp);

  cnt = 0;
  extract(cells, ncell, iso, NULL, 0, &cnt);
  nseg = cnt;
  if (Verbose)
    fprintf(stderr, "cube: nseg: %llu\n", nseg);
  if (nseg == 0) {
    fprintf(stderr, "cube: error: no segments in the mesh\n");
    exit(1);
  }
  if ((tv = (struct Vertex *)malloc(2 * nseg * sizeof *tv)) == NULL) {
    fprintf(stderr, "cube: error: malloc failed\n");
    exit(1);
  }
  cnt = 0;
  extract(cells, ncell, iso, tv, 2 * nseg, &cnt);
  free(cells);

  qsort(tv, 2 * nseg, sizeof *tv, comp_vert);

  cnt = 0;
  createVertexArray(&cnt, tv, 2 * nseg, NULL, 0, NULL);
  nvert = cnt;
  if (Verbose)
    fprintf(stderr, "cube: nvert: %llu\n", nvert);
  if ((vert = (struct Vertex *)malloc(nvert * sizeof *vert)) == NULL) {
    fprintf(stderr, "cube: error: malloc failed\n");
    exit(1);
  }
  if ((seg = (struct int2 *)malloc(nseg * sizeof *seg)) == NULL) {
    fprintf(stderr, "cube: error: malloc failed\n");
    exit(1);
  }
  cnt = 0;
  createVertexArray(&cnt, tv, 2 * nseg, vert, nvert, seg);
  free(tv);

  snprintf(xy_path, sizeof xy_path, "%s.xy.raw", output_path);
  snprintf(seg_path, sizeof seg_path, "%s.seg.raw", output_path);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", output_path);
  snprintf(xdmf_path, sizeof xdmf_path, "%s.xdmf2", output_path);
  xy_base = xy_path;
  seg_base = seg_path;
  attr_base = attr_path;
  for (j = 0; xy_path[j] != '\0'; j++) {
    if (xy_path[j] == '/' && xy_path[j + 1] != '\0') {
      xy_base = &xy_path[j + 1];
      seg_base = &seg_path[j + 1];
      attr_base = &attr_path[j + 1];
    }
  }
  if ((file = fopen(xy_path, "w")) == NULL) {
    fprintf(stderr, "cube: error: fail to open '%s'\n", xy_path);
    exit(1);
  }
  if (ScaleFlag == 0) {
    for (i = 0; i < nvert; i++) {
      xy[0] = vert[i].position.x;
      xy[1] = vert[i].position.y;
      if (fwrite(xy, sizeof xy, 1, file) != 1) {
        fprintf(stderr, "cube: error: fail to write '%s'\n", xy_path);
        exit(1);
      }
    }
  } else {
    double h;
    h = L / (1 << minlevel);
    if (Verbose)
      fprintf(stderr, "cube: X0 Y0 L minlevel: [%g %g] %g %d\n", X0, Y0, L,
              minlevel);
    for (i = 0; i < nvert; i++) {
      xy[0] = vert[i].position.x * h + X0;
      xy[1] = vert[i].position.y * h + Y0;
      if (fwrite(xy, sizeof xy, 1, file) != 1) {
        fprintf(stderr, "cube: error: fail to write '%s'\n", xy_path);
        exit(1);
      }
    }
  }
  if (fclose(file) != 0) {
    fprintf(stderr, "cube: fail to close '%s'\n", xy_path);
    exit(1);
  }
  if ((file = fopen(seg_path, "w")) == NULL) {
    fprintf(stderr, "cube: error: fail to open '%s'\n", seg_path);
    exit(1);
  }
  if (fwrite(seg, nseg * sizeof *seg, 1, file) != 1) {
    fprintf(stderr, "cube: error: fail to write '%s'\n", seg_path);
    exit(1);
  }
  if (fclose(file) != 0) {
    fprintf(stderr, "cube: fail to close '%s'\n", seg_path);
    exit(1);
  }
  if ((attr = (float *)malloc(nvert * sizeof *attr)) == NULL) {
    fprintf(stderr, "cube: error: malloc failed\n");
    exit(1);
  }
  for (i = 0; i < nvert; i++)
    attr[i] = vert[i].field;
  if ((file = fopen(attr_path, "w")) == NULL) {
    fprintf(stderr, "cube: error: fail to open '%s'\n", attr_path);
    exit(1);
  }
  if (fwrite(attr, nvert * sizeof *attr, 1, file) != 1) {
    fprintf(stderr, "cube: error: fail to write '%s'\n", attr_path);
    exit(1);
  }
  if (fclose(file) != 0) {
    fprintf(stderr, "cube: fail to close '%s'\n", attr_path);
    exit(1);
  }
  free(vert);
  free(seg);
  free(attr);

  if ((file = fopen(xdmf_path, "w")) == NULL) {
    fprintf(stderr, "cube: error: fail to open '%s'\n", xdmf_path);
    exit(1);
  }
  fprintf(file,
          "<Xdmf\n"
          "    Version=\"2\">\n"
          "  <Domain>\n"
          "    <Grid>\n"
          "      <Topology\n"
          "         TopologyType=\"Polyline\"\n"
          "         NodesPerElement=\"2\"\n"
          "         Dimensions=\"%llu\">\n"
          "        <DataItem\n"
          "            Dimensions=\"%llu 2\"\n"
          "            NumberType=\"Int\"\n"
          "            Format=\"Binary\">\n"
          "          %s\n"
          "        </DataItem>\n"
          "      </Topology>\n"
          "      <Geometry\n"
          "         GeometryType=\"XY\">\n"
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
  if (fclose(file) != 0) {
    fprintf(stderr, "cube: error: fail to close '%s'\n", xdmf_path);
    exit(1);
  }
}
