.PHONY: install clean
PREFIX = $(HOME)/.local
NVCC = nvcc
NVCCFLAGS = -O2 -g
CC = cc
CFLAGS = -O2 -g
CXXFLAGS =

iso: iso.cu table.inc
	$(NVCC) $(NVCCFLAGS) -Xcompiler '$(CXXFLAGS)' $< -o $@ $(LDFLAGS)

iso-cpu: iso.c table.inc
	$(CC) $(CFLAGS) $< -o $@

cube: cube.c
	$(CC) $(CFLAGS) $< -o $@

iso2d: iso2d.c
	$(CC) $(CFLAGS) $< -o $@ -lm

iso3d: iso3d.c table.inc
	$(CC) $(CFLAGS) $< -o $@ -lm

clean:
	rm -f iso iso-cpu cube iso2d iso3d

install:
	mkdir -p '$(PREFIX)'/bin
	cp iso iso-cpu cube iso2d iso3d '$(PREFIX)'/bin/
