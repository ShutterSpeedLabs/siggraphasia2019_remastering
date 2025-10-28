# Batch Colorization Guide

This script processes multiple videos with their corresponding keyframe references efficiently.

## Folder Structure

```
references/
├── video_1_key_1/     # Keyframes for video 1, segment 1
│   ├── frame001.png
│   ├── frame002.png
│   └── ...
├── video_1_key_2/     # Keyframes for video 1, segment 2
├── video_2_key_1/     # Keyframes for video 2, segment 1
└── ...

input/
├── video_1_1/         # Black & white images for video 1, segment 1
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── video_1_2/         # Black & white images for video 1, segment 2
├── video_2_1/         # Black & white images for video 2, segment 1
└── ...

output/                # Will be created automatically
├── video_1_1/         # Colorized results
├── video_1_2/
└── ...
```

## Usage

### Basic usage (CPU):
```bash
python remaster_batch.py --input_dir input --reference_dir references --output_dir output
```

### With GPU (recommended):
```bash
python remaster_batch.py --input_dir input --reference_dir references --output_dir output --gpu
```

### Custom minimum dimension:
```bash
python remaster_batch.py --input_dir input --reference_dir references --output_dir output --gpu --mindim 320
```

## How It Works

1. The script scans the reference directory for keyframe folders (pattern: `video_X_key_Y`)
2. For each keyframe folder, it finds the matching input folder (pattern: `video_X_Y`)
3. It loads the keyframes into memory
4. Processes all images in the corresponding input folder
5. Saves colorized results to the output folder with the same structure
6. Clears GPU memory before moving to the next folder pair
7. Repeats for all folder pairs

## Memory Optimization

- Processes one folder pair at a time
- Loads keyframes only when needed
- Clears GPU cache after each folder
- Processes images one by one to minimize VRAM usage

## File Name Preservation

- Output files keep the same names as input files
- Folder structure is preserved in the output directory
