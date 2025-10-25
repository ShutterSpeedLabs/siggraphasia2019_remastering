# Memory Optimization Guide for RTX 4060 Ti 16GB

## Problem Summary
The original script ran out of VRAM when processing videos with many reference images (73 refs) because:
- The attention mechanism creates huge matrices: `frames × references × spatial_dimensions`
- With 5 frames and 73 refs at 320p, this requires ~7GB just for the attention computation

## Solutions Implemented

### 1. `remaster_stable.py` ✅ RECOMMENDED - No Flickering
**Best for quality and consistency**

```bash
python remaster_stable.py --input video.mp4 --reference_dir references/ --gpu --block_size 5 --max_refs 20
```

**Features:**
- Samples 20 references evenly from all available references
- Keeps references on GPU for consistency (no CPU-GPU transfers)
- Processes 5 frames at a time
- **No flickering** - all frames use the same reference set
- Speed: ~4.6 fps on RTX 4060 Ti

**Memory Usage:**
- References: ~50MB VRAM
- Processing: ~6-7GB VRAM peak
- Total: ~7-8GB VRAM

### 2. `remaster_adaptive.py` - Chunked References (Has Flickering)
**Uses all 73 references but causes flickering**

```bash
python remaster_adaptive.py --input video.mp4 --reference_dir references/ --gpu
```

**Features:**
- Splits 73 references into chunks (e.g., 5 chunks of 15)
- Processes each chunk separately and averages results
- Stores references in CPU RAM
- Auto-adjusts batch sizes based on available VRAM

**Issues:**
- ⚠️ **Causes flickering** because different chunks produce slightly different colors
- Averaging multiple passes reduces color accuracy

### 3. `remaster_chunked_refs.py` - Manual Chunking
**Similar to adaptive but with manual control**

```bash
python remaster_chunked_refs.py --input video.mp4 --reference_dir references/ --gpu --block_size 5 --ref_chunk_size 15
```

**Features:**
- Manual control over chunk sizes
- Same flickering issue as adaptive version

## Performance Comparison

| Script | References Used | Flickering | Speed | VRAM Usage |
|--------|----------------|------------|-------|------------|
| remaster_stable.py | 20 (sampled) | ❌ No | ~4.6 fps | 7-8 GB |
| remaster_adaptive.py | 73 (chunked) | ⚠️ Yes | ~2.4 fps | 6-8 GB |
| remaster_chunked_refs.py | 73 (chunked) | ⚠️ Yes | ~2.4 fps | 6-8 GB |
| Original remaster.py | All | ❌ No | OOM Error | >16 GB |

## Recommendations

### For Best Quality (No Flickering)
```bash
python remaster_stable.py --input video.mp4 --reference_dir references/ --gpu --block_size 5 --max_refs 20
```

### For Maximum Speed
```bash
python remaster_stable.py --input video.mp4 --reference_dir references/ --gpu --block_size 8 --max_refs 15
```

### For Using More References (30+)
```bash
python remaster_stable.py --input video.mp4 --reference_dir references/ --gpu --block_size 3 --max_refs 30
```

## How Reference Sampling Works

The stable version samples references **evenly** across your collection:
- If you have 73 references and request 20, it takes every 3.65th reference
- This ensures temporal coverage across the entire video
- References are: frame 0, 3, 7, 11, 14, 18, 22, 25, 29, 33, etc.

## Memory Calculation Formula

```
VRAM_needed = model_size + references_size + attention_matrix + frame_processing

Where:
- model_size ≈ 1.2 GB (fixed)
- references_size = num_refs × 3 × 256 × 445 × 4 bytes ≈ 2.5 MB per ref
- attention_matrix = batch_size × num_refs × spatial_size² × 4 bytes
- frame_processing = batch_size × channels × height × width × 4 bytes

For 5 frames, 20 refs, 320p:
- Model: 1.2 GB
- References: 50 MB
- Attention: ~5-6 GB (dominant factor)
- Frames: ~100 MB
- Total: ~7-8 GB
```

## Troubleshooting

### Still Getting OOM Errors?
1. Reduce block_size: `--block_size 3`
2. Reduce max_refs: `--max_refs 15`
3. Reduce resolution: `--mindim 256`

### Want to Use All 73 References Without Flickering?
Not possible with current architecture - the attention mechanism requires too much memory. The chunking approach causes flickering due to averaging.

**Best compromise:** Use 25-30 evenly sampled references with block_size=3

### Processing Too Slow?
1. Increase block_size: `--block_size 8` (if you have VRAM)
2. Reduce max_refs: `--max_refs 15`
3. Check GPU utilization: `nvidia-smi -l 1`

## Output Files

After processing, you'll get:
- `video_2_input_in.mp4` - Grayscale input
- `video_2_input_out.mp4` - Colorized output
- `video_2_input_comp.mp4` - Side-by-side comparison

## Technical Details

### Why Chunking Causes Flickering
When you split references into chunks and average:
```python
# Chunk 1: refs 0-14 → color_output_1
# Chunk 2: refs 15-29 → color_output_2
# Chunk 3: refs 30-44 → color_output_3
# Final = (output_1 + output_2 + output_3) / 3
```

Each chunk produces slightly different colors because:
- Different references emphasize different color palettes
- Averaging reduces saturation and creates inconsistency
- Frame-to-frame variations become visible as flickering

### Why Sampling Works Better
Using a consistent subset of references:
```python
# All frames use refs [0, 3, 7, 11, 14, 18, ...]
# Same references → same color decisions → no flickering
```

The model sees the same reference context for every frame, ensuring temporal consistency.
