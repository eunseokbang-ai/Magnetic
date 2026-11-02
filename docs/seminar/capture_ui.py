"""Full-resolution screenshots of DroneMag Studio for the seminar deck,
driven with Selenium against the running server (port from argv, default
8102): restore the newest autosaved project, grid it, then add the Andong
orthomosaic GeoTIFF and the drill-hole KMZ as reference layers."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

PORT = sys.argv[1] if len(sys.argv) > 1 else "8102"
OUT = Path(__file__).resolve().parent / "fig"
ORTHO = r"D:\Andong20261028\Andong_ortho(2.22.tif"
KMZ = next(Path(r"C:\magnetic\AnDong_Mag").glob("안동시추위치*.kmz"))

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1920,1080")
opts.add_argument("--hide-scrollbars")
opts.add_argument("--force-device-scale-factor=1")
drv = webdriver.Chrome(options=opts)
drv.get(f"http://localhost:{PORT}/")


def buttons(text):
    # textContent, not .text: Selenium's .text is empty for buttons scrolled out of the side panel
    return [b for b in drv.find_elements(By.TAG_NAME, "button") if (b.get_attribute("textContent") or "").strip() == text]


def wait_for(cond, timeout=900, step=1.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            v = cond()
            if v:
                return v
        except Exception:  # noqa: BLE001 - stale elements while React re-renders
            pass
        time.sleep(step)
    raise TimeoutError("timed out")


wait_for(lambda: buttons("복구"))
drv.execute_script("arguments[0].click()", buttons("복구")[0])
print("restoring...", flush=True)
wait_for(lambda: (b := buttons("그리드 생성")) and b[0].is_enabled())
time.sleep(2)
drv.save_screenshot(str(OUT / "ui_restored.png"))
drv.execute_script("arguments[0].scrollIntoView({block: 'center'}); arguments[0].click()", buttons("그리드 생성")[0])
wait_for(lambda: drv.find_elements(By.CSS_SELECTOR, "img.leaflet-image-layer"), timeout=180)
time.sleep(3)
drv.save_screenshot(str(OUT / "ui_grid.png"))
print("grid captured", flush=True)

# reference layers: scroll the layer manager into view, register the GeoTIFF and the KMZ
path_input = wait_for(lambda: [i for i in drv.find_elements(By.CSS_SELECTOR, "input[type=text]") if "GeoTIFF" in (i.get_attribute("placeholder") or "")])[0]
drv.execute_script("arguments[0].scrollIntoView({block: 'center'})", path_input)
path_input.clear()
path_input.send_keys(ORTHO)
drv.execute_script("arguments[0].click()", buttons("GeoTIFF 연동")[0])
wait_for(lambda: [s for s in drv.find_elements(By.CSS_SELECTOR, "span[title]") if "GeoTIFF" in s.text])
file_input = [i for i in drv.find_elements(By.CSS_SELECTOR, "input[type=file]") if ".kmz" in (i.get_attribute("accept") or "")][0]
file_input.send_keys(str(KMZ))
wait_for(lambda: drv.find_elements(By.CSS_SELECTOR, ".leaflet-tooltip"))
# zoom the map in on the survey: scroll-wheel over the grid overlay centre
drv.execute_script("window.scrollTo(0, 0)")
time.sleep(1)
img = drv.find_element(By.CSS_SELECTOR, "img.leaflet-image-layer")
cx, cy = img.rect["x"] + img.rect["width"] / 2, img.rect["y"] + img.rect["height"] / 2
from selenium.webdriver.common.action_chains import ActionChains

for _ in range(2):
    ActionChains(drv).scroll_by_amount(0, -300).perform() if False else None
drv.execute_script("""
const el = document.querySelector('.leaflet-container');
const r = el.getBoundingClientRect();
for (let i = 0; i < 3; i++) el.dispatchEvent(new WheelEvent('wheel', {bubbles: true, clientX: arguments[0], clientY: arguments[1], deltaY: -120}));
""", cx, cy)
time.sleep(6)
drv.save_screenshot(str(OUT / "ui_layers.png"))
print("layers captured", flush=True)
drv.quit()
print("done")
