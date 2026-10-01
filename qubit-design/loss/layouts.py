"""Layouts used in this study (micrometres)."""
from shapely.geometry import box
from shapely.ops import unary_union


def wang_design_A(pad_gap=65.0, stub_w=25.0, stub_len=30.0):
    """Wang et al., APL 107, 162601 (2015), Fig. 1(a): two 250 x 500 um Al pads on
    sapphire in a 3D waveguide cavity, joined at the junction by 25-um-wide stubs.
    The stubs stop stub_len from each pad; the remaining few-um gap holds the <= 1 um
    leads and the junction, which are excluded here as in the paper's "pads" columns."""
    w, L = 250.0, 500.0
    top = unary_union([box(-w / 2, pad_gap / 2, w / 2, pad_gap / 2 + L),
                       box(-stub_w / 2, pad_gap / 2 - stub_len, stub_w / 2, pad_gap / 2)])
    bot = unary_union([box(-w / 2, -pad_gap / 2 - L, w / 2, -pad_gap / 2),
                       box(-stub_w / 2, -pad_gap / 2, stub_w / 2, -pad_gap / 2 + stub_len)])
    return {"conductors": {"pad_top": top, "pad_bot": bot}, "ground": None,
            "chip": (-1000, -2000, 1000, 2000), "t_sub": 430.0, "eps_sub": 10.34,
            "gap_below": 1500.0, "gap_above": 1500.0, "gap_side": 1500.0,
            "mode": {"pad_top": +1.0, "pad_bot": -1.0}, "mode_type": "floating_pair"}


def grounded_transmon(W=24.0, L=620.0, G=30.0, jg=12.0, wc=34.0, lc=121.0, cg=6.0, ws=2.0,
                      eps_sub=11.45, t_sub=525.0, chip_half=900.0, half=True, rounding=0.0):
    """DeviceLayout ExampleRectangleTransmon + ExampleClawedMeanderReadout claw, in Python.

    Island W x L (x in [-W/2, W/2], y in [0, L]) in a ground-plane trench of width G (jg on
    the junction side, y < 0). Readout claw on the far end behind a ws-wide ground shield,
    claw trace wc, finger length lc, claw gap cg; a 10/6 um CPW stub carries the claw up
    (the rest of the resonator is beyond the region that matters for the qubit mode).
    half=True keeps x >= 0 only; the x = 0 plane is then a natural (Neumann) symmetry plane.
    """
    from shapely.geometry import box as B
    from shapely.ops import unary_union as U
    wg = W + 2 * G
    island = B(-W / 2, 0, W / 2, L)
    if rounding > 0:
        island = island.buffer(-rounding).buffer(rounding, quad_segs=6)
    trench = B(-wg / 2, -jg, wg / 2, L + G)
    yt = L + G  # top of trench = inner edge of shield
    hole2 = B(-(wg / 2 + ws + 2 * cg + wc), yt + ws, wg / 2 + ws + 2 * cg + wc, yt + ws + wc + 2 * cg)
    h3x0 = wg / 2 + ws
    hole3 = B(-(h3x0 + 2 * cg + wc), yt + ws - (lc + cg), -h3x0, yt + ws)
    hole4 = B(h3x0, yt + ws - (lc + cg), h3x0 + 2 * cg + wc, yt + ws)
    ytop = yt + ws + wc + 2 * cg
    cpw_hole = B(-11, ytop - 0.01, 11, ytop + 150)
    palm = B(-(wg / 2 + ws + cg + wc), yt + ws + cg, wg / 2 + ws + cg + wc, yt + ws + cg + wc)
    f1 = B(-(wg / 2 + ws + cg + wc), yt + ws - lc, -(wg / 2 + ws + cg), yt + ws + cg)
    f2 = B(wg / 2 + ws + cg, yt + ws - lc, wg / 2 + ws + cg + wc, yt + ws + cg)
    stub = B(-5, yt + ws + cg + wc - 0.01, 5, ytop + 150)
    claw = U([palm, f1, f2, stub])
    cy = (L + 150) / 2
    chip = (-chip_half, cy - chip_half - 200, chip_half, cy + chip_half + 200)
    ground = B(*chip).difference(U([trench, hole2, hole3, hole4, cpw_hole]))
    cond = {"island": island, "claw": claw, "ground": ground}
    if half:
        clip = B(0, chip[1] - 1, chip[2] + 1, chip[3] + 1)
        cond = {k: v.intersection(clip) for k, v in cond.items()}
        chip = (0.0, chip[1], chip[2], chip[3])
    return {"conductors": cond, "ground": "ground", "chip": chip, "t_sub": t_sub, "eps_sub": eps_sub,
            "gap_below": 600.0, "gap_above": 1200.0, "gap_side": 600.0, "half_x": half,
            "mode": {"island": 1.0}, "mode_type": "driven",
            "params": dict(W=W, L=L, G=G, jg=jg, wc=wc, lc=lc, cg=cg, ws=ws, eps_sub=eps_sub, rounding=rounding)}
