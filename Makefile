.PHONY: install
PREFIX = $(HOME)/.local
NVCC = nvcc
NVCCFLAGS = -O2 -g
CXXFLAGS =
iso: iso.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler '$(CXXFLAGS)' $< -o $@ $(LDFLAGS)

clean:
	rm -f iso
install:
	mkdir -p '$(PREFIX)'/bin
	cp iso '$(PREFIX)'/bin/bin/

iso: table.inc
