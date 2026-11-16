"""Cut 2048 x 2048 level-0 crops of tissue from CMU-1.svs (OpenSlide test data, CC0).

crop_cmu1.py SVS OUT_FOLDER [N]

Tissue is found on the slide's 1/16 level (mean RGB below 210); the N non-overlapping crops
with the most tissue are read from level 0 through tifffile's tile store (the slide is never
decoded whole) and saved as zlib RGB TIFFs. The crop origins are printed as JSON.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

svs, out = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SIZE = 2048
out.mkdir(parents=True, exist_ok=True)
with tifffile.TiffFile(svs) as tif:
    level0 = tif.series[0].levels[0]
    low = tif.series[0].levels[2].asarray()  # 1/16 of level 0 for CMU-1
    factor = level0.shape[0] / low.shape[0]
    tissue = low.mean(axis=2) < 210
    step = int(SIZE / factor)
    scores = []
    for y in range(0, tissue.shape[0] - step, step):
        for x in range(0, tissue.shape[1] - step, step):
            scores.append((float(tissue[y : y + step, x : x + step].mean()), y, x))
    scores.sort(reverse=True)
    store = zarr.open_array(tif.pages[0].aszarr(), mode="r")
    origins = []
    for score, y, x in scores[:n]:
        y0, x0 = int(round(y * factor)), int(round(x * factor))
        crop = np.asarray(store[y0 : y0 + SIZE, x0 : x0 + SIZE])
        name = f"CMU-1_y{y0}_x{x0}.tif"
        tifffile.imwrite(out / name, crop, photometric="rgb", compression="zlib")
        origins.append({"file": name, "y0": y0, "x0": x0, "tissue_fraction": score})
print(json.dumps(origins, indent=1))
