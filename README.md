# 3D Gaussian Splatting Training and Viewer

## Features

- **Complete Training Pipeline**: Train 3DGS models from COLMAP reconstructions
- **Multiple Strategies**: Support for both Default and MCMC densification strategies  
- **Real-time Web Viewer**: Interactive 3D visualization via browser using viser/nerfview
- **Evaluation Metrics**: PSNR, SSIM, and LPIPS quality metrics
- **Camera Optimization**: Optional camera pose refinement during training
- **Appearance Modeling**: Per-image appearance embeddings for varying lighting

## Requirements

- **Python**: 3.8+
- **CUDA**: 11.8+ (required for GPU training)
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install gsplat (with CUDA kernels)

```bash
pip install gsplat
```

## Data Preparation

This project expects COLMAP-reconstructed datasets. Your data should be structured as:

```
data/
├── your_scene/
│   ├── images/           # Input images
│   └── sparse/
│       └── 0/            # COLMAP reconstruction
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
```

### Creating a dataset from images

If you have a set of images, you can use COLMAP to create a reconstruction:

```bash
# Install COLMAP first, then:
colmap automatic_reconstructor \
    --workspace_path ./data/your_scene \
    --image_path ./data/your_scene/images
```

### Download sample datasets

```bash
python datasets/download_dataset.py --name garden
```

## Training

### Basic training

```bash
python complete_trainer.py --data_dir data/your_scene --result_dir results/your_scene
```

### Training with evaluation

```bash
python complete_trainer.py \
    --data_dir data/your_scene \
    --result_dir results/your_scene \
    --eval
```

### Full training options

```bash
python complete_trainer.py \
    --data_dir data/your_scene \
    --result_dir results/your_scene \
    --max_steps 30000 \
    --eval \
    --eval_every 100 \
    --sh_degree 3 \
    --data_factor 4 \
    --batch_size 1
```

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | - | Path to COLMAP dataset |
| `--result_dir` | `results/` | Output directory |
| `--max_steps` | 30000 | Training iterations |
| `--data_factor` | 4 | Image downsample factor |
| `--sh_degree` | 3 | Spherical harmonics degree |
| `--eval` | False | Enable evaluation |
| `--eval_every` | 100 | Evaluation frequency |
| `--pose_opt` | False | Enable camera optimization |
| `--app_opt` | False | Enable appearance optimization |

## Viewing Results

### Interactive web viewer

Launch the viewer to visualize trained models or test data:

```bash
# View a trained checkpoint
python viewer.py --ckpt results/your_scene/ckpts/ckpt_29999_rank0.pt --port 8080

# View test data (no checkpoint)
python viewer.py --port 8080
```

Then open your browser and navigate to `http://localhost:8080`.

### Viewer controls

- **Left-click + drag**: Rotate camera
- **Right-click + drag**: Pan camera
- **Scroll**: Zoom in/out
- **Reset**: Double-click to reset view

## Output Structure

After training, results are saved in the following structure:

```
results/your_scene/
├── ckpts/                    # Model checkpoints
│   ├── ckpt_6999_rank0.pt
│   └── ckpt_29999_rank0.pt
├── renders/                  # Rendered images
├── videos/                   # Rendered videos
├── stats/                    # Training statistics
│   └── stats.json
├── metrics.json              # Evaluation metrics
└── cfg.json                  # Training configuration
```

## Metrics

The following metrics are computed during evaluation:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, typically 25-35 dB
- **SSIM** (Structural Similarity Index): Higher is better, 0-1 range
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better, 0-1 range

## Advanced Usage

### MCMC Strategy

Use the MCMC densification strategy for potentially better results:

```bash
python complete_trainer.py \
    --data_dir data/your_scene \
    --result_dir results/your_scene \
    --strategy.type mcmc
```

### Multi-GPU Training

The training script supports distributed training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python complete_trainer.py \
    --data_dir data/your_scene \
    --result_dir results/your_scene
```

### Custom camera trajectories

Generate smooth camera paths for video rendering by providing keyframe poses.

## Troubleshooting

### CUDA Out of Memory

- Reduce `--batch_size` to 1
- Increase `--data_factor` to downsample images more
- Use `--packed` mode for memory-efficient rasterization

### COLMAP not finding images

Ensure your images are in a supported format (JPEG, PNG) and the `sparse/0/` directory contains valid COLMAP files.

### Viewer not loading

- Check that the checkpoint file exists
- Ensure the port is not in use
- Try a different browser (Chrome/Firefox recommended)