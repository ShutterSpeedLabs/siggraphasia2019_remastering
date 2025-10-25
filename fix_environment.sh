#!/bin/bash

echo "Fixing PyTorch environment for CUDA 13.0 / RTX 4060 Ti..."
echo ""
echo "Current PyTorch version:"
conda run -n remaster python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo ""
echo "This will update PyTorch to a compatible version..."
echo ""

# Update PyTorch to a version compatible with CUDA 11.8 (closest to 13.0)
conda install -n remaster pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo ""
echo "Updated PyTorch version:"
conda run -n remaster python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
echo ""
echo "Environment fixed! You can now run the optimized script."
