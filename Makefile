NVCC = nvcc
NVCCFLAGS = -O2 -g
iso: iso.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f iso
