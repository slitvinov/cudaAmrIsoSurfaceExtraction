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

clean:
	rm -f iso iso-cpu cube

install:
	mkdir -p '$(PREFIX)'/bin
	cp iso iso-cpu cube '$(PREFIX)'/bin/
