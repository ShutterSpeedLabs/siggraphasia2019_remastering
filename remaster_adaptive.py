"""
   Adaptive memory-optimized version for RTX 4060 Ti 16GB
   - Dynamically adjusts batch sizes based on available GPU memory
   - Uses CPU RAM for all reference images
   - Processes references in chunks to avoid OOM
   - Maximizes throughput while staying within memory limits
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

parser = argparse.ArgumentParser(description='Remastering - Adaptive Memory')
parser.add_argument('--input',   type=str,   default='none', help='Input video')
parser.add_argument('--reference_dir',  type=str, default='none', help='Path to the reference image directory')
parser.add_argument('--disable_colorization', action='store_true', default=False, help='Remaster without colorization')
parser.add_argument('--gpu',       action='store_true', default=False, help='Use GPU')
parser.add_argument('--mindim',     type=int,   default='320',    help='Length of minimum image edges')
parser.add_argument('--block_size', type=int,   default='10',    help='Number of frames to process at once (auto-adjusted)')
parser.add_argument('--ref_chunk_size', type=int, default='20', help='Number of reference images per chunk')
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
    
    # Adjust block size based on available memory
    # Conservative settings to avoid OOM with attention mechanism
    if mem_free > 12:
        opt.block_size = 8
        opt.ref_chunk_size = 15
        print(f'High memory available: Using block_size={opt.block_size}, ref_chunk_size={opt.ref_chunk_size}')
    elif mem_free > 10:
        opt.block_size = 6
        opt.ref_chunk_size = 12
        print(f'Good memory available: Using block_size={opt.block_size}, ref_chunk_size={opt.ref_chunk_size}')
    elif mem_free > 8:
        opt.block_size = 5
        opt.ref_chunk_size = 10
        print(f'Moderate memory available: Using block_size={opt.block_size}, ref_chunk_size={opt.ref_chunk_size}')
    else:
        opt.block_size = 3
        opt.ref_chunk_size = 8
        print(f'Low memory available: Using block_size={opt.block_size}, ref_chunk_size={opt.ref_chunk_size}')

print('Processing %s...'%os.path.basename(opt.input))

outputdir = 'tmp/'
outputdir_in = outputdir+'input/'
os.makedirs( outputdir_in, exist_ok=True )
outputdir_out = outputdir+'output/'
os.makedirs( outputdir_out, exist_ok=True )

# Prepare reference images - KEEP ALL IN CPU RAM
refimgs_chunks = []
if not opt.disable_colorization:
   if opt.reference_dir!='none':
      import glob
      ext_list = ['png','jpg','bmp']
      reference_files = []
      for ext in ext_list:
         reference_files += glob.glob( opt.reference_dir+'/*.'+ext, recursive=True )
      
      print(f'Found {len(reference_files)} reference images')
      print(f'Loading all references into CPU RAM...')
      
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
      
      # Split references into chunks to fit in VRAM during processing
      chunk_size = opt.ref_chunk_size
      num_chunks = (len(reference_files) + chunk_size - 1) // chunk_size
      
      print(f'Splitting {len(reference_files)} references into {num_chunks} chunks of ~{chunk_size}')
      
      for chunk_idx in range(num_chunks):
         start_idx = chunk_idx * chunk_size
         end_idx = min(start_idx + chunk_size, len(reference_files))
         chunk_refs = refs[start_idx:end_idx]
         
         chunk_tensor = torch.FloatTensor(len(chunk_refs), 3, target_h, target_w)
         for i, v in enumerate(chunk_refs):
            refimg = utils.addMergin( v, target_w=target_w, target_h=target_h )
            chunk_tensor[i] = transforms.ToTensor()( refimg )
         
         chunk_tensor = chunk_tensor.view(1, chunk_tensor.size(0), chunk_tensor.size(1), chunk_tensor.size(2), chunk_tensor.size(3))
         # Keep in CPU RAM - pin memory for faster transfers
         if opt.gpu:
            chunk_tensor = chunk_tensor.pin_memory()
         refimgs_chunks.append(chunk_tensor)
      
      # Calculate CPU RAM usage
      cpu_ram_mb = sum([chunk.element_size() * chunk.nelement() for chunk in refimgs_chunks]) / 1024**2
      print(f'All {len(reference_files)} references loaded in CPU RAM ({cpu_ram_mb:.1f} MB)')
      print(f'Will process in {num_chunks} GPU chunks to avoid OOM')

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
print(f'Processing {block} frames at a time')

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
      input_frames = input_frames.to(device, non_blocking=True)
      if gtC is not None:
         gtC = gtC.to(device, non_blocking=True)

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
      # Perform colorization with chunked references
      else:
         if opt.reference_dir=='none':
            output_ab = modelC( output_l )
         else:
            # Process each reference chunk and average the results
            output_ab_accumulated = None
            
            for chunk_idx, refimgs_chunk in enumerate(refimgs_chunks):
               # Move this chunk from CPU to GPU (non_blocking for speed)
               refimgs_gpu = refimgs_chunk.to(device, non_blocking=True)
               
               # Process with this chunk
               output_ab_chunk = modelC( output_l, refimgs_gpu )
               
               # Accumulate results
               if output_ab_accumulated is None:
                  output_ab_accumulated = output_ab_chunk
               else:
                  output_ab_accumulated = output_ab_accumulated + output_ab_chunk
               
               # Free GPU memory immediately
               del refimgs_gpu, output_ab_chunk
               if opt.gpu:
                  torch.cuda.empty_cache()
            
            # Average the accumulated results
            output_ab = output_ab_accumulated / len(refimgs_chunks)
            del output_ab_accumulated
         
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
   
   print("Done!")
