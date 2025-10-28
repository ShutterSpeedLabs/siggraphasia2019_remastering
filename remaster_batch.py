"""
   Batch colorization script for multiple videos with keyframe references
   Based on the working remaster_stable.py code
   
   Folder structure:
   - references/video_X_key_Y/ - contains reference keyframes
   - input/video_X_Y/ - contains black and white images to colorize
   - output/video_X_Y/ - will contain colorized results
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse
import glob
import re
import cv2
import warnings
import utils

# Suppress LAB->RGB conversion warnings (expected when colors are outside RGB gamut)
warnings.filterwarnings('ignore', message='.*CIE-LAB.*')

parser = argparse.ArgumentParser(description='Batch Remastering with Keyframes')
parser.add_argument('--input_dir', type=str, default='input', help='Input directory containing video folders')
parser.add_argument('--reference_dir', type=str, default='references', help='Reference directory containing keyframe folders')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory for colorized images')
parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU')
parser.add_argument('--mindim', type=int, default=320, help='Length of minimum image edges')
parser.add_argument('--batch_size', type=int, default=5, help='Number of images to process together (temporal batch)')
parser.add_argument('--overlap', type=int, default=2, help='Number of overlapping frames between batches for temporal consistency')
opt = parser.parse_args()

device = torch.device('cuda:0' if opt.gpu else 'cpu')

# Clear GPU cache if using GPU
if opt.gpu:
    torch.cuda.empty_cache()

print("Loading models...")
# Load remaster network
modelR = __import__('model.remasternet', fromlist=['NetworkR']).NetworkR()
state_dict = torch.load('model/remasternet.pth.tar', map_location=device, weights_only=False)
modelR.load_state_dict(state_dict['modelR'])
modelR = modelR.to(device)
modelR.eval()

modelC = __import__('model.remasternet', fromlist=['NetworkC']).NetworkC()
modelC.load_state_dict(state_dict['modelC'])
modelC = modelC.to(device)
modelC.eval()

print("Models loaded successfully!")

def get_matching_pairs(reference_dir, input_dir):
    """
    Match keyframe folders with input folders
    Returns list of tuples: (keyframe_folder, input_folder)
    """
    pairs = []
    
    # Get all keyframe folders (e.g., video_1_key_1, video_2_key_1)
    keyframe_folders = sorted([d for d in os.listdir(reference_dir) 
                               if os.path.isdir(os.path.join(reference_dir, d)) and '_key_' in d])
    
    for keyframe_folder in keyframe_folders:
        # Extract pattern: video_X_key_Y -> video_X_Y
        match = re.match(r'video_(\d+)_key_(\d+)', keyframe_folder)
        if match:
            video_num = match.group(1)
            key_num = match.group(2)
            input_folder = f'video_{video_num}_{key_num}'
            
            input_path = os.path.join(input_dir, input_folder)
            if os.path.isdir(input_path):
                pairs.append((keyframe_folder, input_folder))
            else:
                print(f"Warning: No matching input folder found for {keyframe_folder} (expected {input_folder})")
    
    return pairs

def load_reference_images(reference_path, device):
    """Load all reference images from a keyframe folder"""
    ext_list = ['png', 'jpg', 'bmp', 'jpeg']
    reference_files = []
    for ext in ext_list:
        reference_files += glob.glob(os.path.join(reference_path, f'*.{ext}'))
    
    reference_files = sorted(reference_files)
    
    if not reference_files:
        print(f"Warning: No reference images found in {reference_path}")
        return None
    
    print(f"  Loading {len(reference_files)} reference images...")
    
    aspect_mean = 0
    refs = []
    for v in reference_files:
        refimg = Image.open(v).convert('RGB')
        w, h = refimg.size
        aspect_mean += w/h
        refs.append(refimg)
    
    aspect_mean /= len(reference_files)
    target_w = int(256*aspect_mean) if aspect_mean > 1 else 256
    target_h = 256 if aspect_mean >= 1 else int(256/aspect_mean)
    
    refimgs = torch.FloatTensor(len(reference_files), 3, target_h, target_w)
    for i, refimg in enumerate(refs):
        refimg = utils.addMergin(refimg, target_w=target_w, target_h=target_h)
        refimgs[i] = transforms.ToTensor()(refimg)
    
    refimgs = refimgs.view(1, refimgs.size(0), refimgs.size(1), refimgs.size(2), refimgs.size(3)).to(device)
    return refimgs

def process_images_batch(img_paths, refimgs, modelR, modelC, device, mindim):
    """Process multiple images together with temporal dimension - like video processing"""
    if not img_paths:
        return []
    
    # Load and prepare all frames
    frames_l = []
    original_sizes = []
    
    for img_path in img_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error loading {img_path}")
            continue
        
        h, w = frame.shape[:2]
        original_sizes.append((w, h))
        
        # Resize if needed
        minwh = min(w, h)
        scale = 1
        if minwh != mindim:
            scale = mindim / minwh
        t_w = int(round(w * scale / 16.) * 16)
        t_h = int(round(h * scale / 16.) * 16)
        
        if t_w != w or t_h != h:
            frame = cv2.resize(frame, (t_w, t_h))
        
        # Convert to grayscale
        frame_l = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_l = torch.from_numpy(frame_l).view(frame_l.shape[0], frame_l.shape[1], 1)
        frame_l = frame_l.permute(2, 0, 1).float() / 255.0
        frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
        frames_l.append(frame_l)
    
    if not frames_l:
        return []
    
    # Concatenate along temporal dimension (like video processing)
    input_frames = torch.cat(frames_l, 2)  # [1, 1, T, H, W]
    input_frames = input_frames.to(device)
    
    with torch.no_grad():
        # Restoration
        output_l = modelR(input_frames)
        
        # Colorization with reference images
        output_ab = modelC(output_l, refimgs)
        
        # Move to CPU
        output_l_cpu = output_l.cpu()
        output_ab_cpu = output_ab.cpu()
        
        # Free GPU memory
        del output_l, output_ab, input_frames
        if opt.gpu:
            torch.cuda.empty_cache()
        
        # Convert each frame back to RGB
        output_imgs = []
        for i in range(len(frames_l)):
            out_l = output_l_cpu[0, :, i, :, :]
            out_c = output_ab_cpu[0, :, i, :, :]
            output = torch.cat((out_l, out_c), dim=0).numpy().transpose((1, 2, 0))
            output_rgb = utils.convertLAB2RGB(output)
            output_img = Image.fromarray(np.uint8(output_rgb * 255))
            output_imgs.append(output_img)
    
    return output_imgs

def colorize_folder(keyframe_folder, input_folder, reference_dir, input_dir, output_dir, 
                   modelR, modelC, device, mindim):
    """Colorize all images in an input folder using keyframes from reference folder"""
    print(f"\nProcessing: {input_folder} with keyframes from {keyframe_folder}")
    
    # Load reference images
    reference_path = os.path.join(reference_dir, keyframe_folder)
    refimgs = load_reference_images(reference_path, device)
    
    if refimgs is None:
        print(f"  Skipping {input_folder} - no reference images")
        return
    
    # Get input images
    input_path = os.path.join(input_dir, input_folder)
    ext_list = ['png', 'jpg', 'bmp', 'jpeg']
    input_files = []
    for ext in ext_list:
        input_files += glob.glob(os.path.join(input_path, f'*.{ext}'))
    
    input_files = sorted(input_files)
    
    if not input_files:
        print(f"  No input images found in {input_path}")
        return
    
    # Create output directory
    output_path = os.path.join(output_dir, input_folder)
    os.makedirs(output_path, exist_ok=True)
    
    print(f"  Colorizing {len(input_files)} images in batches of {opt.batch_size} (overlap={opt.overlap})...")
    
    # Process images in overlapping batches for temporal consistency
    batch_size = opt.batch_size
    overlap = min(opt.overlap, batch_size // 2)  # Overlap shouldn't be more than half batch
    stride = batch_size - overlap
    
    pbar = tqdm(total=len(input_files), desc=f"  {input_folder}")
    processed_count = 0
    
    i = 0
    while i < len(input_files):
        # Get batch with overlap
        batch_end = min(i + batch_size, len(input_files))
        batch_files = input_files[i:batch_end]
        
        # Process batch
        output_imgs = process_images_batch(batch_files, refimgs, modelR, modelC, device, mindim)
        
        # Determine which frames to save (skip overlapping frames from previous batch)
        if i == 0:
            # First batch: save all
            save_start = 0
        else:
            # Skip overlap frames (already saved from previous batch)
            save_start = overlap
        
        # Save outputs (only non-overlapping frames)
        for j in range(save_start, len(output_imgs)):
            img_file = batch_files[j]
            output_img = output_imgs[j]
            filename = os.path.basename(img_file)
            output_file = os.path.join(output_path, filename)
            output_img.save(output_file)
            processed_count += 1
        
        pbar.update(len(output_imgs) - save_start)
        
        # Move to next batch
        if i == 0:
            i += batch_size  # First batch: full stride
        else:
            i += stride  # Subsequent batches: stride with overlap
    
    pbar.close()
    
    # Clear GPU cache after processing each folder
    if opt.gpu:
        torch.cuda.empty_cache()
    
    print(f"  ✓ Completed {input_folder}")

# Main processing
def main():
    print(f"\nScanning directories...")
    print(f"  Reference dir: {opt.reference_dir}")
    print(f"  Input dir: {opt.input_dir}")
    print(f"  Output dir: {opt.output_dir}")
    
    # Get matching pairs
    pairs = get_matching_pairs(opt.reference_dir, opt.input_dir)
    
    if not pairs:
        print("\nNo matching folder pairs found!")
        print("Expected structure:")
        print("  references/video_X_key_Y/ - keyframe folders")
        print("  input/video_X_Y/ - input image folders")
        return
    
    print(f"\nFound {len(pairs)} folder pairs to process:")
    for keyframe_folder, input_folder in pairs:
        print(f"  {keyframe_folder} -> {input_folder}")
    
    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # Process each pair
    print("\n" + "="*60)
    print("Starting batch colorization...")
    print("="*60)
    
    for keyframe_folder, input_folder in pairs:
        colorize_folder(
            keyframe_folder, input_folder,
            opt.reference_dir, opt.input_dir, opt.output_dir,
            modelR, modelC, device, opt.mindim
        )
    
    print("\n" + "="*60)
    print("Batch colorization completed!")
    print(f"Results saved to: {opt.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
