#include <stdint.h>
#define __constant__ static const
#include "table.inc"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

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
  struct vec3i u;
  u.x = v.x >> s;
  u.y = v.y >> s;
  u.z = v.z >> s;
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
                    struct Vertex *out, int size, unsigned long long *cnt) {
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
        if (out == NULL || id >= (unsigned long long)(3 * size))
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
    vert[id].position.x = vertex.position.x;
    vert[id].position.y = vertex.position.y;
    vert[id].position.z = vertex.position.z;
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
  if (vec3f_lt(a->position, b->position))
    return -1;
  if (vec3f_lt(b->position, a->position))
    return 1;
  return 0;
}

int main(int argc, char **argv) {
  double X0, Y0, Z0, L;
  float iso, *attr, xyz[3];
  struct int3 *tri;
  int Verbose, maxlevel;
  long j, size;
  FILE *file, *cell_file, *scalar_file, *field_file;
  int cell[4], ox, oy, oz, minlevel, ScaleFlag;
  char attr_path[FILENAME_MAX], xyz_path[FILENAME_MAX], tri_path[FILENAME_MAX],
      xdmf_path[FILENAME_MAX], *attr_base, *xyz_base, *tri_base, *cell_path,
      *scalar_path, *field_path, *output_path, *end;
  struct Cell *cells;
  struct Vertex *tv, *vert;
  unsigned long long nvert, ntri, ncell, cnt, i;

  (void)argc;
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
    if (cells[i].level > maxlevel)
      maxlevel = cells[i].level;
    if (cells[i].lower.x < ox)
      ox = cells[i].lower.x;
    if (cells[i].lower.y < oy)
      oy = cells[i].lower.y;
    if (cells[i].lower.z < oz)
      oz = cells[i].lower.z;
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

  cnt = 0;
  extract(cells, ncell, iso, NULL, 0, &cnt);
  ntri = cnt;
  if (Verbose)
    fprintf(stderr, "iso: ntri: %llu\n", ntri);
  if (ntri == 0) {
    fprintf(stderr, "iso: error: no triangles in the mesh\n");
    exit(1);
  }
  if ((tv = (struct Vertex *)malloc(3 * ntri * sizeof *tv)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  cnt = 0;
  extract(cells, ncell, iso, tv, 3 * ntri, &cnt);
  free(cells);

  qsort(tv, 3 * ntri, sizeof *tv, comp_vert);

  cnt = 0;
  createVertexArray(&cnt, tv, 3 * ntri, NULL, 0, NULL);
  nvert = cnt;
  if (Verbose)
    fprintf(stderr, "iso: nvert: %llu\n", nvert);
  if ((vert = (struct Vertex *)malloc(nvert * sizeof *vert)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  if ((tri = (struct int3 *)malloc(ntri * sizeof *tri)) == NULL) {
    fprintf(stderr, "iso: error: malloc failed\n");
    exit(1);
  }
  cnt = 0;
  createVertexArray(&cnt, tv, 3 * ntri, vert, nvert, tri);
  free(tv);

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
