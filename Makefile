NVCC = nvcc
NVCCFLAGS = -O2 -g
CXXFLAGS =
iso: iso.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler '$(CXXFLAGS)' $< -o $@ $(LDFLAGS)

clean:
	rm -f iso
