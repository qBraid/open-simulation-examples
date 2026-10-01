"""Make headless copies of the published reference codes.
Only two kinds of edit: remove plotting, and save the final design after the main loop.
The cantilever copy also swaps in the boundary conditions of Sigmund (2001), Sec. 5."""
from pathlib import Path


def edit(src, dst, subs, after_last_end=None):
    lines = Path(src).read_text().splitlines()
    for old, new in subs:
        hits = [i for i, l in enumerate(lines) if l.strip() == old]
        assert len(hits) == 1, (src, old, hits)
        lines[hits[0]] = new
    if after_last_end:
        i = max(k for k, l in enumerate(lines) if l.strip() == "end" and not l.startswith(" "))
        lines.insert(i + 1, after_last_end)
    Path(dst).write_text("\n".join(lines) + "\n")


edit("top88.m", "top88_ref.m", [
    ("function top88(nelx,nely,volfrac,penal,rmin,ft)", "function top88_ref(nelx,nely,volfrac,penal,rmin,ft)"),
    ("colormap(gray); imagesc(1-xPhys); caxis([0 1]); axis equal; axis off; drawnow;", "  % (plotting removed for headless runs)"),
], after_last_end='dlmwrite(sprintf("out/ref_top88_mbb_%dx%d_ft%d.txt",nelx,nely,ft), xPhys, "precision", "%.10g");')

edit("top88_ref.m", "top88_cant_ref.m", [
    ("function top88_ref(nelx,nely,volfrac,penal,rmin,ft)", "function top88_cant_ref(nelx,nely,volfrac,penal,rmin,ft)"),
    ("F = sparse(2,1,-1,2*(nely+1)*(nelx+1),1);", "F = sparse(2*(nely+1)*(nelx+1),1,-1,2*(nely+1)*(nelx+1),1);"),
    ("fixeddofs = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]);", "fixeddofs = [1:2*(nely+1)];"),
    ('dlmwrite(sprintf("out/ref_top88_mbb_%dx%d_ft%d.txt",nelx,nely,ft), xPhys, "precision", "%.10g");',
     'dlmwrite(sprintf("out/ref_top88_cant_%dx%d_ft%d.txt",nelx,nely,ft), xPhys, "precision", "%.10g");'),
])

edit("top3d.m", "top3d_ref.m", [
    ("function top3d(nelx,nely,nelz,volfrac,penal,rmin)", "function top3d_ref(nelx,nely,nelz,volfrac,penal,rmin)"),
    ("clf; display_3D(xPhys);", 'dlmwrite(sprintf("out/ref_top3d_%dx%dx%d.txt",nelx,nely,nelz), xPhys(:), "precision", "%.10g");'),
])

src = Path("top3d.m").read_text().splitlines()
a = next(i for i, l in enumerate(src) if l.startswith("function [KE] = lk_H8"))
b = next(i for i, l in enumerate(src) if i > a and l.startswith("% === DISPLAY"))
Path("lk_H8.m").write_text("\n".join(src[a:b]) + "\n")
print("headless reference copies written")
