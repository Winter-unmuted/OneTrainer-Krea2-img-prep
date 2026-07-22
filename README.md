# Krea2 Crop Tool

Point-and-click crop prep for Krea2 LoRA training in [OneTrainer](https://github.com/Nerogar/OneTrainer).

Cut your images to exact aspect-bucket sizes so OneTrainer never resizes them at train time. You decide what gets cropped, not a center-crop.

## Why

OneTrainer sorts images into aspect-ratio buckets and scales/crops each one to fit. Krea2's buckets are all multiples of 64. If an image already matches a bucket exactly, the scale/crop step is a no-op. So: pre-cut everything to real bucket dimensions and point one concept per resolution at the output. Zero resize, no surprise crops, and you control the framing.

## What it does

**Place exact-size crop boxes.** Wheel to cycle bucket sizes, click to drop a box. The box is a real bucket resolution (e.g. 896x1152), cut verbatim, no resampling.

<img width="2248" height="1394" alt="image" src="https://github.com/user-attachments/assets/d853e83b-8654-4e0f-a895-58f78d001b2b" />

**Snap to bucket.** Downscale the image so the whole thing fills the closest bucket with minimal loss. Arrows / Ctrl+←→ step through the best fits, `>` marks the one you're on.

<img width="1015" height="1003" alt="image" src="https://github.com/user-attachments/assets/d4a5e6de-4354-4166-a896-7ef43b1d0077" />

**Rotate / flip.** 90° turns and H/V flips, applied to the pixels (WYSIWYG). Clones off if the image already has boxes.

**Bucket-colored sizes.** Blue = smallest target, red = largest. Same colors in the size list and the snap menu.

<img width="325" height="880" alt="image" src="https://github.com/user-attachments/assets/b3964b7b-69fa-4c9c-b58d-1e2246f59bde" />

**Output ready for OneTrainer.** Crops go to `crops/512`, `crops/768`, `crops/1024` etc., one folder per target family, mixed dimensions inside. Output field takes a path, not just a name: `crops`, `crops/train`, `../shared/crops`, or an absolute path. Change it per batch to split runs.

**Bucket tally.** Counts each size on disk and, given your batch size, shows batches formed and images dropped per epoch (OneTrainer drops the remainder of any bucket that isn't a multiple of batch size, and skips buckets smaller than one batch).

<img width="837" height="1001" alt="image" src="https://github.com/user-attachments/assets/0bd0b83f-5eef-442d-8505-8be7521bebd2" />

**Similarity sort + exclude.** Sort the file list by color-histogram similarity (size- and crop-tolerant), seeded from the biggest image so clusters fall next to each other. Mark junk/dupes as excluded (Ctrl+T) and they drop to the bottom of every sort. Excluded images hold their crop boxes untouched and are skipped by Crop / Crop All until you un-exclude them.

<img width="714" height="751" alt="image" src="https://github.com/user-attachments/assets/d8c9f6bb-6f87-4f44-b717-d76e24e82434" />

**Quality report.** Four measured signals per image, not an LLM guessing: sharpness (blur), color cast, clipping, JPEG blockiness. Sortable table with thumbnails, worst-percentile rows flagged red, batch-exclude the flagged ones, double-click a row to jump straight to that image. Also usable as a file-list sort mode (worst-first).

<img width="1346" height="1001" alt="image" src="https://github.com/user-attachments/assets/9f3392be-bd85-4af0-955d-12906ec2b7aa" />

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

**One concept per output folder**, Resolution Override **on** for each:

- path `crops/512`, Resolution Override on, value `512`
- path `crops/768`, Resolution Override on, value `768`
- path `crops/1024`, Resolution Override on, value `1024`

<img width="805" height="547" alt="image" src="https://github.com/user-attachments/assets/9a99d95b-dff6-4464-93e7-e17e2f42a6df" />


Aspect bucketing on, crop jitter off. Each concept accepts mixed sizes and routes every pre-cut image to its own bucket at scale 1.0. Don't put a comma list on one folder or the random per-image target roll will downscale things.

Note that each concept size is the total number of pixels, so multiple aspect ratios will be in each concept. You can inspect this with the concept statistics in OneTrainer.

<img width="1426" height="804" alt="image" src="https://github.com/user-attachments/assets/724daed9-adb3-40a7-8924-37bf2fa15152" />

## Run

```
pip install pillow numpy
python krea2_crop_tool.py
```

Pick your resolutions (e.g. `512, 768, 1024`) and a folder. Crops land in `<folder>/crops/`.
Note you can include subdirectories, or start in a subdirectory and save up levels with ../ in the file path box.

numpy is only needed for similarity sort and the quality report; everything else runs on Pillow alone. Windows / tkinter. Reads EXIF orientation. Exports JPEG q100 4:4:4 with the source filename in the comment.
