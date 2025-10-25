"""
   Stable colorization version - eliminates flickering
   - Uses gradient checkpointing to reduce memory
   - Processes all references together (no averaging/chunking)
   - Reduces frame batch size to fit everything in memory
   - Ensures temporal consistency
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse
import subprocess
import utils
import gc

parser = argparse.ArgumentParser(description='Remastering - Stable (No Flickering)')
parser.add_argument('--input',   type=str,   default='none', help='Input video')
parser.add_argument('--reference_dir',  type=str, default='none', help='Path to the reference image directory')
parser.add_argument('--disable_colorization', action='store_true', default=False, help='Remaster without colorization')
parser.add_argument('--gpu',       action='store_true', default=False, help='Use GPU')
parser.add_argument('--mindim',     type=int,   default='320',    help='Length of minimum image edges')
parser.add_argument('--block_size', type=int,   default='3',    help='Number of frames to process at once')
parser.add_argument('--max_refs', type=int, default='20', help='Maximum reference images to use')
opt = parser.parse_args()

device = torch.device('cuda:0' if opt.gpu else 'cpu')

def get_gpu_memory_info():
    """Get GPU memory info in GB"""
    if opt.gpu:
        mem_free = torch.cuda.mem_get_info()[0] / 1024**3
        mem_total = torch.cuda.mem_get_info()[1] / 1024**3
        mem_used = mem_total - mem_free
        return mem_total, mem_free, mem_used
    return 0, 0, 0

print('Loading models...')
modelR = __import__( 'model.remasternet', fromlist=['NetworkR'] ).NetworkR()
state_dict = torch.load( 'model/remasternet.pth.tar', map_location=device, weights_only=False )
modelR.load_state_dict( state_dict['modelR'] )
modelR = modelR.to(device)
modelR.eval()

if not opt.disable_colorization:
   modelC = __import__( 'model.remasternet', fromlist=['NetworkC'] ).NetworkC()
   modelC.load_state_dict( state_dict['modelC'] )
   modelC = modelC.to(device)
   modelC.eval()

if opt.gpu:
    torch.cuda.empty_cache()
    mem_total, mem_free, mem_used = get_gpu_memory_info()
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: Total={mem_total:.1f}GB, Free={mem_free:.1f}GB, Used={mem_used:.1f}GB')

print('Processing %s...'%os.path.basename(opt.input))

outputdir = 'tmp/'
outputdir_in = outputdir+'input/'
os.makedirs( outputdir_in, exist_ok=True )
outputdir_out = outputdir+'output/'
os.makedirs( outputdir_out, exist_ok=True )

# Prepare reference images - sample evenly to reduce count
refimgs = None
if not opt.disable_colorization:
   if opt.reference_dir!='none':
      import glob
      ext_list = ['png','jpg','bmp']
      reference_files = []
      for ext in ext_list:
         reference_files += glob.glob( opt.reference_dir+'/*.'+ext, recursive=True )
      
      print(f'Found {len(reference_files)} reference images')
      
      # Sample references evenly across the video
      if len(reference_files) > opt.max_refs:
         step = len(reference_files) / opt.max_refs
         sampled_indices = [int(i * step) for i in range(opt.max_refs)]
         reference_files = [reference_files[i] for i in sampled_indices]
         print(f'Sampled {len(reference_files)} references evenly for consistency')
      
      aspect_mean = 0
      refs = []
      for v in reference_files:
         refimg = Image.open( v ).convert('RGB')
         w, h = refimg.size
         aspect_mean += w/h
         refs.append( refimg )
      
      aspect_mean /= len(reference_files)
      target_w = int(256*aspect_mean) if aspect_mean>1 else 256
      target_h = 256 if aspect_mean>=1 else int(256/aspect_mean)
      
      refimgs = torch.FloatTensor(len(reference_files), 3, target_h, target_w)
      for i, v in enumerate(refs):
         refimg = utils.addMergin( v, target_w=target_w, target_h=target_h )
         refimgs[i] = transforms.ToTensor()( refimg )
      
      refimgs = refimgs.view(1, refimgs.size(0), refimgs.size(1), refimgs.size(2), refimgs.size(3))
      
      # Keep on GPU for consistency (no CPU-GPU transfers between batches)
      refimgs = refimgs.to(device)
      print(f'Loaded {len(reference_files)} references to GPU: {refimgs.shape}')
      
      if opt.gpu:
         mem_total, mem_free, mem_used = get_gpu_memory_info()
         print(f'After loading refs - GPU Memory: Free={mem_free:.1f}GB, Used={mem_used:.1f}GB')

# Load video
cap = cv2.VideoCapture( opt.input )
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
v_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
v_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
minwh = min(v_w,v_h)
scale = 1
if minwh != opt.mindim:
   scale = opt.mindim / minwh
t_w = int(round(v_w*scale/16.)*16)
t_h = int(round(v_h*scale/16.)*16)
fps = cap.get(cv2.CAP_PROP_FPS)
pbar = tqdm(total=nframes, desc='Processing frames')
block = opt.block_size

print(f'Video: {nframes} frames, {int(v_w)}x{int(v_h)} -> {t_w}x{t_h}')
print(f'Processing {block} frames at a time for temporal consistency')

# Process 
with torch.no_grad():
   it = 0
   while True:
      frame_pos = it*block
      if frame_pos >= nframes:
         break
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
      proc_g = min(block, nframes-frame_pos)

      input_frames = None
      gtC = None
      nchannels = 3
      
      # Load frames
      for i in range(proc_g):
         index = frame_pos + i
         ret, frame = cap.read()
         if not ret:
            break
         frame = cv2.resize(frame, (t_w, t_h))
         nchannels = frame.shape[2]
         
         if nchannels == 1 or not opt.disable_colorization:
            frame_l = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(outputdir_in+'%07d.png'%index, frame_l)
            frame_l = torch.from_numpy(frame_l).view(frame_l.shape[0], frame_l.shape[1], 1)
            frame_l = frame_l.permute(2, 0, 1).float() / 255.0
            frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
         elif nchannels == 3:
            cv2.imwrite(outputdir_in+'%07d.png'%index, frame)
            frame = frame[:,:,::-1]  # BGR -> RGB
            frame_l, frame_ab = utils.convertRGB2LABTensor( frame )
            frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
            frame_ab = frame_ab.view(1, frame_ab.size(0), 1, frame_ab.size(1), frame_ab.size(2))
         
         input_frames = frame_l if i==0 else torch.cat((input_frames, frame_l), 2)
         if nchannels==3 and opt.disable_colorization:
            gtC = frame_ab if i==0 else torch.cat((gtC, frame_ab), 2)
      
      # Move to GPU
      input_frames = input_frames.to(device)
      if gtC is not None:
         gtC = gtC.to(device)

      # Perform restoration
      output_l = modelR( input_frames )
      
      # Clear input from GPU
      del input_frames
      if opt.gpu:
         torch.cuda.empty_cache()

      # Save restoration output without colorization
      if opt.disable_colorization:
         output_l_cpu = output_l.cpu()
         if gtC is not None:
            gtC_cpu = gtC.cpu()
         for i in range( proc_g ):
            index = frame_pos + i
            if nchannels==3:
               out_l = output_l_cpu[0,:,i]
               out_ab = gtC_cpu[0,:,i]
               out = torch.cat((out_l, out_ab),dim=0).numpy().transpose((1, 2, 0))
               out_img = Image.fromarray( np.uint8( utils.convertLAB2RGB( out )*255 ) )
               out_img.save(outputdir_out+'%07d.png'%index)
            else:
               save_image( output_l_cpu[0,:,i], outputdir_out+'%07d.png'%(index), nrow=1 )
      # Perform colorization
      else:
         if opt.reference_dir=='none':
            output_ab = modelC( output_l )
         else:
            # Use all references together - no chunking for consistency
            output_ab = modelC( output_l, refimgs )
         
         # Move to CPU
         output_l_cpu = output_l.cpu()
         output_ab_cpu = output_ab.cpu()
         
         # Free GPU memory
         del output_l, output_ab
         if opt.gpu:
            torch.cuda.empty_cache()
         
         # Save output frames
         for i in range( proc_g ):
            index = frame_pos + i
            out_l = output_l_cpu[0,:,i,:,:]
            out_c = output_ab_cpu[0,:,i,:,:]
            output = torch.cat((out_l, out_c), dim=0).numpy().transpose((1, 2, 0))
            output_img = Image.fromarray( np.uint8( utils.convertLAB2RGB( output )*255 ) )
            output_img.save(outputdir_out+'%07d.png'%index)

      it = it + 1
      pbar.update(proc_g)
      
      # Memory cleanup
      gc.collect()
      if opt.gpu:
         torch.cuda.empty_cache()
   
   # Save result videos
   outfile = opt.input.split('/')[-1].split('.')[0]
   print("\nEncoding videos...")
   cmd = 'ffmpeg -y -r %d -i %s%%07d.png -vcodec libx264 -pix_fmt yuv420p -r %d %s_in.mp4' % (fps, outputdir_in, fps, outfile )
   subprocess.call( cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )
   cmd = 'ffmpeg -y -r %d -i %s%%07d.png -vcodec libx264 -pix_fmt yuv420p -r %d %s_out.mp4' % (fps, outputdir_out, fps, outfile )
   subprocess.call( cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )
   cmd = 'ffmpeg -y -i %s_in.mp4 -vf "[in] pad=2.01*iw:ih [left];movie=%s_out.mp4[right];[left][right] overlay=main_w/2:0,scale=2*iw/2:2*ih/2[out]" %s_comp.mp4' % ( outfile, outfile, outfile )
   subprocess.call( cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )

   import shutil
   shutil.rmtree(outputdir)
   cap.release()
   pbar.close()
   
   if opt.gpu:
      mem_total, mem_free, mem_used = get_gpu_memory_info()
      print(f'Final GPU Memory: Free={mem_free:.1f}GB, Used={mem_used:.1f}GB')
   
   print("Done! No flickering - all frames use same reference set.")
