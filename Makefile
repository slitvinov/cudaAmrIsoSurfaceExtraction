iso: iso.cu
	$(NVCC) -std=c++14 -O3 $< -o $@

clean:
	rm -f iso
