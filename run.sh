#!/bin/sh

set -e

make iso-cpu cube iso2d iso3d

python gen3d.py
./iso-cpu -v in.cells in.scalar in.field 0 mesh
python view3d.py mesh

python gen3d-amr.py
./iso-cpu -v in.cells in.scalar in.field 0 mesh-amr
python view3d.py mesh-amr

python gen2d.py
./cube -v in.cells in.scalar in.field 0 mesh2d
python view2d.py mesh2d in.cells

python gen2d-amr.py
./cube -v in.cells in.scalar in.field 0 mesh2d-amr
python view2d.py mesh2d-amr in.cells

cp mesh.svg img/gen3d.svg
cp mesh-amr.svg img/gen3d-amr.svg
cp mesh2d.svg img/gen2d.svg
cp mesh2d-amr.svg img/gen2d-amr.svg

for xdmf in output/*.xdmf2; do
    base=${xdmf%.xdmf2}
    xyz=${base}.xyz.raw
    attr=${base}.attr.raw
    [ -f "$xyz" ] && [ -f "$attr" ] || continue
    sz=$(wc -c < "$xyz")
    [ $((sz % 96)) -eq 0 ] || continue
    out=${base}-iso
    echo "=== $base ==="
    ./iso3d -v "$xyz" "$attr:0:6" "$attr:0:6" 0.5 "$out"
    python viewtri.py "$out"
done
