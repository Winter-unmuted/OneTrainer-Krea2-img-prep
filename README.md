# Krea2 Crop Tool

Point-and-click crop prep for Krea2 LoRA training in [OneTrainer](https://github.com/Nerogar/OneTrainer).

Cut your images to exact aspect-bucket sizes so OneTrainer never resizes them at train time. You decide what gets cropped, not a center-crop.

## Why

OneTrainer sorts images into aspect-ratio buckets and scales/crops each one to fit. Krea2's buckets are all multiples of 64. If an image already matches a bucket exactly, the scale/crop step is a no-op. So: pre-cut everything to real bucket dimensions and point one concept per resolution at the output. Zero resize, no surprise crops, and you control the framing.

## What it does

**Place exact-size crop boxes.** Wheel to cycle bucket sizes, click to drop a box. The box is a real bucket resolution (e.g. 896x1152), cut verbatim, no resampling.

![place boxes](docs/place.png)

**Snap to bucket.** Downscale the image so the whole thing fills the closest bucket with minimal loss. Arrows / Ctrl+←→ step through the best fits, `>` marks the one you're on.

![snap](docs/snap.png)

**Rotate / flip.** 90° turns and H/V flips, applied to the pixels (WYSIWYG, no orientation metadata tricks). Clones off instead of altering the image if it already has boxes.

**Bucket-colored sizes.** Blue = smallest target, red = largest. Same colors in the size list and the snap-to-bucket menu.

![colors](docs/colors.png)

**Output ready for OneTrainer.** Crops go to `crops/512`, `crops/768`, `crops/1024` etc., one folder per target family, mixed dimensions inside. Output field takes a path, not just a name: `crops`, `crops/train`, `../shared/crops`, or an absolute path. Change it per batch to split runs.

**Bucket tally.** Counts each size on disk and, given your batch size, shows batches formed and images dropped per epoch (OneTrainer drops the remainder of any bucket that isn't a multiple of batch size, and skips buckets smaller than one batch).

![tally](docs/tally.png)

**Similarity sort + exclude.** Sort the file list by color-histogram similarity (size- and crop-tolerant), seeded from the biggest image so clusters fall next to each other. Mark junk/dupes as excluded (Ctrl+T) and they drop to the bottom of every sort. Excluded images hold their crop boxes untouched and are skipped by Crop / Crop All until you un-exclude them.

![similarity](docs/similarity.png)

**Quality report.** Four measured signals per image, not an LLM guessing: sharpness (blur), color cast, clipping, JPEG blockiness. Sortable table with thumbnails, worst-percentile rows flagged red, batch-exclude the flagged ones, double-click a row to jump straight to that image. Also usable as a file-list sort mode (worst-first).

![quality report](docs/quality.png)

Plus: dark mode, Open folder / Reload without restarting, optional recursive subfolder scan, sortable file list (name / megapixels / crop count / similarity / quality), Crop All, per-session save/resume, no-upscale hard rule.

## Hotkeys

| Key | Action |
| --- | --- |
| Mouse wheel (over image) | cycle crop size |
| Left click | place box |
| Right click | select / deselect box |
| Arrows | nudge selected box 1px (Shift = 16px) |
| Delete / Backspace | delete selected box |
| Esc | deselect box |
| PgUp / PgDn | previous / next image |
| C or Ctrl+E | crop this image |
| Ctrl+D | clone + downscale |
| Ctrl+← / → | step to prev / next best-fit bucket |
| Ctrl+↑ / ↓ | jump to prev / next image with crops |
| Alt+↑ / ↓ | jump to prev / next image without crops |
| Ctrl+T | exclude / include (trash) — also works in the Quality report |

## OneTrainer setup

One concept per output folder, Resolution Override **on** for each:

- path `crops/512`, Resolution Override on, value `512`
- path `crops/768`, Resolution Override on, value `768`
- path `crops/1024`, Resolution Override on, value `1024`

Aspect bucketing on, crop jitter off. Each concept accepts mixed sizes and routes every pre-cut image to its own bucket at scale 1.0. Don't put a comma list on one folder or the random per-image target roll will downscale things.

## Run

```
pip install pillow numpy
python Onetrainer_crop_prep.py
```

Pick your resolutions (e.g. `512, 768, 1024`) and a folder. Crops land in `<folder>/crops/`.

numpy is only needed for similarity sort and the quality report; everything else runs on Pillow alone. Windows / tkinter. Reads EXIF orientation. Exports JPEG q100 4:4:4 with the source filename in the comment.
