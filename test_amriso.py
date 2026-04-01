import numpy as np
import amriso


def test_openmp():
    assert hasattr(amriso, 'openmp')
    assert amriso.openmp in (0, 1)
    assert hasattr(amriso, 'openmp_max_threads')
    assert amriso.openmp_max_threads >= 1


def test_example2d():
    coords, scalar = amriso.example2d()
    assert coords.shape == (400, 4, 2)
    assert scalar.shape == (400,)


def test_example3d():
    coords, scalar = amriso.example3d()
    assert coords.shape == (1000, 8, 3)
    assert scalar.shape == (1000,)


def test_extract2d():
    coords, scalar = amriso.example2d()
    xy, seg, attr = amriso.extract2d(coords, scalar, scalar, 0.0)
    assert xy.shape[1] == 2
    assert seg.shape[1] == 2
    assert len(attr) == len(xy)
    assert len(seg) == 32
    assert len(xy) == 32


def test_extract3d():
    coords, scalar = amriso.example3d()
    xyz, tri, attr = amriso.extract3d(coords, scalar, scalar, 0.0)
    assert xyz.shape[1] == 3
    assert tri.shape[1] == 3
    assert len(attr) == len(xyz)
    assert len(tri) == 128
    assert len(xyz) == 66


def test_workspace2d():
    coords, scalar = amriso.example2d()
    xy1, seg1, attr1 = amriso.extract2d(coords, scalar, scalar, 0.0)

    nc = len(scalar)
    work = np.empty(amriso.workspace_size2d(nc), dtype=np.uint8)
    nmax = 4 * nc
    xy = np.empty((nmax, 2), dtype=np.float32)
    seg = np.empty((nmax, 2), dtype=np.int32)
    attr = np.empty(nmax, dtype=np.float32)
    ns, nv = amriso.extract2d(coords, scalar, scalar, 0.0,
                               out=(xy, seg, attr), work=work)
    assert nv == len(xy1)
    assert ns == len(seg1)
    assert np.allclose(np.sort(xy1, axis=0), np.sort(xy[:nv], axis=0))


def test_workspace3d():
    coords, scalar = amriso.example3d()
    xyz1, tri1, attr1 = amriso.extract3d(coords, scalar, scalar, 0.0)

    nc = len(scalar)
    work = np.empty(amriso.workspace_size3d(nc), dtype=np.uint8)
    nmax = 5 * nc
    xyz = np.empty((nmax, 3), dtype=np.float32)
    tri = np.empty((nmax, 3), dtype=np.int32)
    attr = np.empty(nmax, dtype=np.float32)
    nt, nv = amriso.extract3d(coords, scalar, scalar, 0.0,
                               out=(xyz, tri, attr), work=work)
    assert nv == len(xyz1)
    assert nt == len(tri1)
    assert np.allclose(np.sort(xyz1, axis=0), np.sort(xyz[:nv], axis=0))


def test_empty():
    coords, scalar = amriso.example2d()
    xy, seg, attr = amriso.extract2d(coords, scalar, scalar, 1000.0)
    assert len(xy) == 0
    assert len(seg) == 0


def test_dump2d(tmp_path):
    coords, scalar = amriso.example2d()
    xy, seg, attr = amriso.extract2d(coords, scalar, scalar, 0.0)
    prefix = str(tmp_path / "test2d")
    amriso.dump2d(prefix, xy, seg, attr)
    assert (tmp_path / "test2d.xdmf2").exists()
    assert (tmp_path / "test2d.xy.raw").exists()
    assert (tmp_path / "test2d.seg.raw").exists()
    assert (tmp_path / "test2d.attr.raw").exists()


def test_dump3d(tmp_path):
    coords, scalar = amriso.example3d()
    xyz, tri, attr = amriso.extract3d(coords, scalar, scalar, 0.0)
    prefix = str(tmp_path / "test3d")
    amriso.dump3d(prefix, xyz, tri, attr)
    assert (tmp_path / "test3d.xdmf2").exists()
    assert (tmp_path / "test3d.xyz.raw").exists()
    assert (tmp_path / "test3d.tri.raw").exists()
    assert (tmp_path / "test3d.attr.raw").exists()


if __name__ == "__main__":
    import sys
    passed = 0
    failed = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            import inspect
            if 'tmp_path' in inspect.signature(func).parameters:
                import tempfile, pathlib
                with tempfile.TemporaryDirectory() as d:
                    func(pathlib.Path(d))
            else:
                func()
            print("  PASS", name)
            passed += 1
        except Exception as e:
            print("  FAIL", name, e)
            failed += 1
    print("%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
