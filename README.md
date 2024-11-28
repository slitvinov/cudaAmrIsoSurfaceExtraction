# cudaAmrIsoSurfaceExtraction

Sample code for our CUDA AMR Dual-Mesh Generation / Iso-Surface
Extraction Paper

Wald, Ingo. "A simple, general, and GPU friendly method for computing
dual mesh and iso-surfaces of adaptive mesh refinement (AMR) data."
arXiv preprint arXiv:2004.08475 (2020).
<https://arxiv.org/abs/2004.08475>

# Run

```
$ make
nvcc -O2 -g -Xcompiler '' iso.cu -o iso
$ python gen.py
gen.py: in.cell in.scalar in.field
$ ./iso -v in.cells in.scalar in.field 0 mesh
```
