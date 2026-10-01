"""Verify the viewer the way the Agent Canvas shows it (srcdoc, sandbox=allow-scripts, origin null),
and take the README screenshots. Waits on window.__ready instead of a fixed time budget.

Usage: python verify_viewer.py viewer.html out_dir
"""
import sys, html, os, time
from playwright.sync_api import sync_playwright

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
page_html = open(src).read()
STATES = [  # (name, theme, hash) — hash drives the viewer's deep links
    ("viewer_light", "light", "lead=0&ov=tcwv&mode=fc&cam=-55,20,4.8"),
    ("viewer_dark_laura_swipe", "dark", "lead=72&ov=msl&mode=swipe&cam=-88,25,2.0"),
    ("viewer_error_day5", "light", "lead=120&ov=t2m&mode=err&cam=20,-35,3.4&details=1"),
    ("viewer_dark_bavi", "dark", "lead=60&ov=tcwv&mode=fc&cam=127,31,2.4"),
]
args = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
with sync_playwright() as p:
    b = p.chromium.launch(args=args)
    # 1) sandboxed srcdoc, exactly like the canvas
    pg = b.new_page(viewport={"width": 1280, "height": 820})
    errs = []
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.set_content('<!doctype html><body style="margin:0"><iframe id=f sandbox="allow-scripts" style="border:0;width:100vw;height:100vh" srcdoc="'
                   + html.escape(page_html, quote=True) + '"></iframe>')
    fr = pg.frame_locator("#f")
    t0 = time.time()
    inner = pg.frames[1]
    inner.wait_for_function("window.__ready === true", timeout=120000)
    ready_s = time.time() - t0
    pg.wait_for_timeout(2500)
    inner.evaluate("window.__freeze = true"); inner.wait_for_function("window.__frozen === true", timeout=60000)
    pg.screenshot(timeout=120000, path=os.path.join(out, "viewer_sandbox.png"))
    print(f"sandbox: ready in {ready_s:.1f}s, origin={inner.evaluate('location.origin')}, errors={errs[:3]}")
    # 2) README states (direct file, deep links)
    for name, theme, h in STATES:
        pg2 = b.new_page(viewport={"width": 1440, "height": 900}, color_scheme=theme)
        pg2.goto("file://" + os.path.abspath(src) + "#" + h + "&theme=" + theme)
        pg2.wait_for_function("window.__ready === true", timeout=120000)
        pg2.wait_for_timeout(3500)
        pg2.evaluate("window.__freeze = true"); pg2.wait_for_function("window.__frozen === true", timeout=60000)
        pg2.screenshot(timeout=120000, path=os.path.join(out, name + ".png"))
        pg2.close()
        print("shot", name)
    b.close()
