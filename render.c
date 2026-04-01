#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t viridis[][3] = {
    {68, 1, 84},    {72, 35, 116},  {64, 67, 135},  {52, 94, 141},
    {41, 120, 142}, {32, 144, 140}, {34, 167, 132}, {66, 190, 113},
    {122, 209, 81}, {189, 222, 38}, {253, 231, 37}, {253, 231, 37},
};

struct v3 {
  float x, y, z;
};

static struct v3 v3sub(struct v3 a, struct v3 b) {
  struct v3 r = {a.x - b.x, a.y - b.y, a.z - b.z};
  return r;
}
static struct v3 v3cross(struct v3 a, struct v3 b) {
  struct v3 r = {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x};
  return r;
}
static float v3dot(struct v3 a, struct v3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static float v3len(struct v3 a) { return sqrtf(v3dot(a, a)); }
static struct v3 v3norm(struct v3 a) {
  float l = v3len(a);
  if (l > 0) {
    a.x /= l;
    a.y /= l;
    a.z /= l;
  }
  return a;
}

static void colormap(float t, uint8_t *r, uint8_t *g, uint8_t *b) {
  int n = sizeof viridis / sizeof viridis[0] - 1;
  float s;
  int i;
  if (t <= 0) {
    *r = viridis[0][0];
    *g = viridis[0][1];
    *b = viridis[0][2];
    return;
  }
  if (t >= 1) {
    *r = viridis[n][0];
    *g = viridis[n][1];
    *b = viridis[n][2];
    return;
  }
  s = t * n;
  i = (int)s;
  s -= i;
  *r = (uint8_t)(viridis[i][0] * (1 - s) + viridis[i + 1][0] * s);
  *g = (uint8_t)(viridis[i][1] * (1 - s) + viridis[i + 1][1] * s);
  *b = (uint8_t)(viridis[i][2] * (1 - s) + viridis[i + 1][2] * s);
}

static float edgef(float ax, float ay, float bx, float by, float cx,
                   float cy) {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

int main(int argc, char **argv) {
  int W, H, Verbose, have_attr, up;
  char *prefix, *output;
  char xyz_path[1024], tri_path[1024], attr_path[1024];
  FILE *ftri;
  float *verts, *attrs;
  float *zbuf;
  uint8_t *pixels;
  long vsize, asize;
  int32_t nvert, ntri;
  float lo[3], hi[3], center[3], extent;
  float elev, azim;
  float cosA, sinA, cosE, sinE;
  struct v3 light;
  float attr_lo, attr_hi;

  (void)argc;
  W = 800;
  H = 600;
  Verbose = 0;
  elev = 30;
  azim = 45;
  up = 2;
  while (*++argv != NULL && argv[0][0] == '-')
    switch (argv[0][1]) {
    case 'h':
      fprintf(stderr,
              "Usage: render [-v] [-s WxH] [-e elev] [-a azim] [-u y|z] "
              "prefix output.png\n");
      exit(0);
    case 'v':
      Verbose = 1;
      break;
    case 's':
      argv++;
      if (!argv[0] || sscanf(argv[0], "%dx%d", &W, &H) != 2) {
        fprintf(stderr, "render: error: -s needs WxH\n");
        exit(1);
      }
      break;
    case 'e':
      argv++;
      if (!argv[0]) {
        fprintf(stderr, "render: error: -e needs angle\n");
        exit(1);
      }
      elev = (float)atof(argv[0]);
      break;
    case 'a':
      argv++;
      if (!argv[0]) {
        fprintf(stderr, "render: error: -a needs angle\n");
        exit(1);
      }
      azim = (float)atof(argv[0]);
      break;
    case 'u':
      argv++;
      if (!argv[0] || (argv[0][0] != 'y' && argv[0][0] != 'z')) {
        fprintf(stderr, "render: error: -u needs y or z\n");
        exit(1);
      }
      up = (argv[0][0] == 'y') ? 1 : 2;
      break;
    default:
      fprintf(stderr, "render: error: unknown option '%s'\n", *argv);
      exit(1);
    }
  if (!argv[0] || !argv[1]) {
    fprintf(stderr,
            "Usage: render [-v] [-s WxH] [-e elev] [-a azim] [-u y|z] prefix "
            "output.png\n");
    exit(1);
  }
  prefix = argv[0];
  output = argv[1];

  snprintf(xyz_path, sizeof xyz_path, "%s.xyz.raw", prefix);
  snprintf(tri_path, sizeof tri_path, "%s.tri.raw", prefix);
  snprintf(attr_path, sizeof attr_path, "%s.attr.raw", prefix);

  {
    FILE *f = fopen(xyz_path, "rb");
    if (!f) {
      fprintf(stderr, "render: error: cannot open '%s'\n", xyz_path);
      exit(1);
    }
    fseek(f, 0, SEEK_END);
    vsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    nvert = (int32_t)(vsize / (3 * sizeof(float)));
    verts = malloc(vsize);
    if (!verts) {
      fprintf(stderr, "render: error: malloc failed\n");
      exit(1);
    }
    if (fread(verts, 1, vsize, f) != (size_t)vsize) {
      fprintf(stderr, "render: error: short read '%s'\n", xyz_path);
      exit(1);
    }
    fclose(f);
  }

  {
    FILE *f = fopen(attr_path, "rb");
    have_attr = (f != NULL);
    if (have_attr) {
      fseek(f, 0, SEEK_END);
      asize = ftell(f);
      fseek(f, 0, SEEK_SET);
      attrs = malloc(asize);
      if (!attrs) {
        fprintf(stderr, "render: error: malloc failed\n");
        exit(1);
      }
      if (fread(attrs, 1, asize, f) != (size_t)asize) {
        fprintf(stderr, "render: error: short read '%s'\n", attr_path);
        exit(1);
      }
      fclose(f);
    } else {
      attrs = NULL;
    }
  }

  {
    FILE *f = fopen(tri_path, "rb");
    if (!f) {
      fprintf(stderr, "render: error: cannot open '%s'\n", tri_path);
      exit(1);
    }
    fseek(f, 0, SEEK_END);
    ntri = (int32_t)(ftell(f) / (3 * sizeof(int32_t)));
    fclose(f);
  }

  lo[0] = lo[1] = lo[2] = 1e30f;
  hi[0] = hi[1] = hi[2] = -1e30f;
  attr_lo = 1e30f;
  attr_hi = -1e30f;
  {
    int32_t i;
    for (i = 0; i < nvert; i++) {
      float x = verts[3 * i], y = verts[3 * i + 1], z = verts[3 * i + 2];
      if (x < lo[0])
        lo[0] = x;
      if (y < lo[1])
        lo[1] = y;
      if (z < lo[2])
        lo[2] = z;
      if (x > hi[0])
        hi[0] = x;
      if (y > hi[1])
        hi[1] = y;
      if (z > hi[2])
        hi[2] = z;
      if (have_attr) {
        if (attrs[i] < attr_lo)
          attr_lo = attrs[i];
        if (attrs[i] > attr_hi)
          attr_hi = attrs[i];
      }
    }
  }
  center[0] = 0.5f * (lo[0] + hi[0]);
  center[1] = 0.5f * (lo[1] + hi[1]);
  center[2] = 0.5f * (lo[2] + hi[2]);
  extent = hi[0] - lo[0];
  if (hi[1] - lo[1] > extent)
    extent = hi[1] - lo[1];
  if (hi[2] - lo[2] > extent)
    extent = hi[2] - lo[2];
  extent *= 0.6f;

  if (Verbose)
    fprintf(stderr,
            "render: nvert=%d ntri=%d bbox=[%.3f,%.3f]x[%.3f,%.3f]x[%.3f,%.3f]"
            " size=%dx%d\n",
            nvert, ntri, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], W, H);

  cosA = cosf(azim * (float)M_PI / 180);
  sinA = sinf(azim * (float)M_PI / 180);
  cosE = cosf(elev * (float)M_PI / 180);
  sinE = sinf(elev * (float)M_PI / 180);
  light = v3norm((struct v3){0.3f, 0.5f, 0.8f});

  zbuf = malloc((size_t)W * H * sizeof *zbuf);
  pixels = malloc((size_t)W * H * 3);
  if (!zbuf || !pixels) {
    fprintf(stderr, "render: error: malloc failed\n");
    exit(1);
  }
  {
    int32_t p;
    for (p = 0; p < W * H; p++) {
      zbuf[p] = 1e30f;
      pixels[3 * p] = pixels[3 * p + 1] = pixels[3 * p + 2] = 240;
    }
  }

  ftri = fopen(tri_path, "rb");
  if (!ftri) {
    fprintf(stderr, "render: error: cannot open '%s'\n", tri_path);
    exit(1);
  }

  {
    int32_t ti, idx[3];
    float scale = (W < H ? W : H) / (2.0f * extent);
    for (ti = 0; ti < ntri; ti++) {
      struct v3 p[3], n;
      float sx[3], sy[3], sz[3];
      float fa[3];
      float shade, diff;
      uint8_t cr, cg, cb;
      int32_t j, xmin, xmax, ymin, ymax, px, py;

      if (fread(idx, sizeof idx, 1, ftri) != 1)
        break;
      for (j = 0; j < 3; j++) {
        float x = verts[3 * idx[j]] - center[0];
        float y = verts[3 * idx[j] + 1] - center[1];
        float z = verts[3 * idx[j] + 2] - center[2];
        float rx, ry, rz, tmp;
        if (up == 1) {
          tmp = y;
          y = z;
          z = tmp;
        }
        rx = x * cosA - y * sinA;
        ry = x * sinA + y * cosA;
        rz = z;
        float ex = rx;
        float ey = ry * cosE - rz * sinE;
        float ez = ry * sinE + rz * cosE;
        p[j] = (struct v3){ex, ey, ez};
        sx[j] = W / 2.0f + ex * scale;
        sy[j] = H / 2.0f - ez * scale;
        sz[j] = ey;
        fa[j] = have_attr ? attrs[idx[j]] : 0.5f;
      }
      n = v3norm(v3cross(v3sub(p[1], p[0]), v3sub(p[2], p[0])));
      if (n.y > 0)
        n = (struct v3){-n.x, -n.y, -n.z};
      diff = v3dot(n, light);
      if (diff < 0)
        diff = 0;
      shade = 0.3f + 0.7f * diff;

      xmin = (int)floorf(sx[0]);
      xmax = (int)ceilf(sx[0]);
      ymin = (int)floorf(sy[0]);
      ymax = (int)ceilf(sy[0]);
      for (j = 1; j < 3; j++) {
        int t;
        t = (int)floorf(sx[j]);
        if (t < xmin)
          xmin = t;
        t = (int)ceilf(sx[j]);
        if (t > xmax)
          xmax = t;
        t = (int)floorf(sy[j]);
        if (t < ymin)
          ymin = t;
        t = (int)ceilf(sy[j]);
        if (t > ymax)
          ymax = t;
      }
      if (xmin < 0)
        xmin = 0;
      if (ymin < 0)
        ymin = 0;
      if (xmax >= W)
        xmax = W - 1;
      if (ymax >= H)
        ymax = H - 1;

      {
        float area =
            edgef(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
        if (fabsf(area) < 1e-6f)
          continue;
        for (py = ymin; py <= ymax; py++)
          for (px = xmin; px <= xmax; px++) {
            float fx = px + 0.5f, fy = py + 0.5f;
            float w0 = edgef(sx[1], sy[1], sx[2], sy[2], fx, fy);
            float w1 = edgef(sx[2], sy[2], sx[0], sy[0], fx, fy);
            float w2 = edgef(sx[0], sy[0], sx[1], sy[1], fx, fy);
            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) ||
                (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
              float z, a;
              int32_t pi;
              w0 /= area;
              w1 /= area;
              w2 /= area;
              z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
              pi = py * W + px;
              if (z < zbuf[pi]) {
                float rv, gv, bv;
                zbuf[pi] = z;
                a = w0 * fa[0] + w1 * fa[1] + w2 * fa[2];
                if (have_attr && attr_hi > attr_lo)
                  colormap((a - attr_lo) / (attr_hi - attr_lo), &cr, &cg, &cb);
                else {
                  cr = 100;
                  cg = 149;
                  cb = 237;
                }
                rv = cr * shade;
                gv = cg * shade;
                bv = cb * shade;
                pixels[3 * pi] = (uint8_t)(rv > 255 ? 255 : rv);
                pixels[3 * pi + 1] = (uint8_t)(gv > 255 ? 255 : gv);
                pixels[3 * pi + 2] = (uint8_t)(bv > 255 ? 255 : bv);
              }
            }
          }
      }
    }
  }
  fclose(ftri);

  if (!stbi_write_png(output, W, H, 3, pixels, W * 3)) {
    fprintf(stderr, "render: error: failed to write '%s'\n", output);
    exit(1);
  }
  if (Verbose)
    fprintf(stderr, "render: wrote '%s'\n", output);

  free(verts);
  free(attrs);
  free(zbuf);
  free(pixels);
}
