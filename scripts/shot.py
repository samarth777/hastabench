import asyncio, sys, os
from playwright.async_api import async_playwright
async def main():
    url = "file://" + os.path.abspath("site/index.html")
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--allow-file-access-from-files"])
        pg = await b.new_page(viewport={"width": 1280, "height": 900})
        errs = []
        pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        await pg.goto(url); await pg.wait_for_timeout(5000)
        os.makedirs("scratch", exist_ok=True); await pg.screenshot(path="scratch/site_full.png", full_page=True)
        for i, sel in enumerate(["#fig1", "#grid", "#fig4", "#stage", ".replay"]):
            el = await pg.query_selector(sel)
            if el:
                await el.scroll_into_view_if_needed(); await pg.wait_for_timeout(800)
                await el.screenshot(path=f"scratch/site_{i}.png")
        print("\n".join(errs) or "no console errors")
        await b.close()
asyncio.run(main())
