"""
3D Gaussian Splatting Viewer

An interactive web-based viewer for 3D Gaussian Splatting models.
Supports both .pt checkpoint files and .ply point cloud files.

Usage:
    # View a trained checkpoint
    python viewer.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt --port 8080

    # View a PLY file
    python viewer.py --ply output.ply --port 8080

    # View test data (no checkpoint)
    python viewer.py --port 8080
"""

import argparse
import math
import os
import time
from typing import Optional, Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

from gsplat.rendering import rasterization


def load_ply(ply_path: str, device: torch.device) -> dict:
    """
    Load a 3D Gaussian Splatting PLY file.
    
    Supports the standard 3DGS PLY format with:
    - positions (x, y, z)
    - normals (nx, ny, nz)
    - spherical harmonics (f_dc_*, f_rest_*)
    - opacity
    - scale (scale_0, scale_1, scale_2)
    - rotation quaternion (rot_0, rot_1, rot_2, rot_3)
    """
    from plyfile import PlyData
    
    print(f"Loading PLY file: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Extract positions
    x = np.array(vertex['x'], dtype=np.float32)
    y = np.array(vertex['y'], dtype=np.float32)
    z = np.array(vertex['z'], dtype=np.float32)
    means = np.stack([x, y, z], axis=-1)
    
    # Extract opacities (stored as raw values, need sigmoid)
    if 'opacity' in vertex.data.dtype.names:
        opacities = np.array(vertex['opacity'], dtype=np.float32)
    else:
        opacities = np.ones(len(x), dtype=np.float32) * 0.5
    
    # Extract scales (stored as log, need exp)
    if 'scale_0' in vertex.data.dtype.names:
        scale_0 = np.array(vertex['scale_0'], dtype=np.float32)
        scale_1 = np.array(vertex['scale_1'], dtype=np.float32)
        scale_2 = np.array(vertex['scale_2'], dtype=np.float32)
        scales = np.stack([scale_0, scale_1, scale_2], axis=-1)
    else:
        scales = np.ones((len(x), 3), dtype=np.float32) * 0.01
    
    # Extract rotations (quaternion wxyz or xyzw - need to check format)
    if 'rot_0' in vertex.data.dtype.names:
        rot_0 = np.array(vertex['rot_0'], dtype=np.float32)
        rot_1 = np.array(vertex['rot_1'], dtype=np.float32)
        rot_2 = np.array(vertex['rot_2'], dtype=np.float32)
        rot_3 = np.array(vertex['rot_3'], dtype=np.float32)
        # Standard 3DGS format: rot_0 is w, rot_1-3 are xyz
        quats = np.stack([rot_0, rot_1, rot_2, rot_3], axis=-1)
    else:
        quats = np.zeros((len(x), 4), dtype=np.float32)
        quats[:, 0] = 1.0  # Identity quaternion
    
    # Extract spherical harmonics for colors
    # DC component (first 3 SH coefficients for RGB)
    sh_coeffs = []
    if 'f_dc_0' in vertex.data.dtype.names:
        f_dc_0 = np.array(vertex['f_dc_0'], dtype=np.float32)
        f_dc_1 = np.array(vertex['f_dc_1'], dtype=np.float32)
        f_dc_2 = np.array(vertex['f_dc_2'], dtype=np.float32)
        sh0 = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=-1)
        sh_coeffs.append(sh0)
        
        # Higher order SH (f_rest_*)
        rest_names = sorted([n for n in vertex.data.dtype.names if n.startswith('f_rest_')])
        if rest_names:
            rest_coeffs = []
            for name in rest_names:
                rest_coeffs.append(np.array(vertex[name], dtype=np.float32))
            shN = np.stack(rest_coeffs, axis=-1)
            
            # Reshape to (N, num_sh, 3) format
            # Original 3DGS stores as (N, 3 * (sh_degree+1)^2 - 3)
            num_rest = len(rest_names)
            if num_rest % 3 == 0:
                shN = shN.reshape(-1, num_rest // 3, 3)
            else:
                # Fallback: just use DC
                shN = None
        else:
            shN = None
    elif 'red' in vertex.data.dtype.names:
        # Simple RGB colors
        r = np.array(vertex['red'], dtype=np.float32) / 255.0
        g = np.array(vertex['green'], dtype=np.float32) / 255.0
        b = np.array(vertex['blue'], dtype=np.float32) / 255.0
        # Convert RGB to SH0
        C0 = 0.28209479177387814
        sh0 = (np.stack([r, g, b], axis=-1) - 0.5) / C0
        sh_coeffs.append(sh0)
        shN = None
    else:
        # Default white
        C0 = 0.28209479177387814
        sh0 = np.ones((len(x), 3), dtype=np.float32) * 0.5 / C0
        sh_coeffs.append(sh0)
        shN = None
    
    # Convert to tensors
    means = torch.from_numpy(means).float().to(device)
    quats = torch.from_numpy(quats).float().to(device)
    quats = F.normalize(quats, p=2, dim=-1)
    
    # Handle scales (stored as log in some formats)
    scales = torch.from_numpy(scales).float().to(device)
    if scales.min() < -10:  # Likely log-scale
        scales = torch.exp(scales)
    
    # Handle opacities (stored as logit in some formats)
    opacities = torch.from_numpy(opacities).float().to(device)
    if opacities.min() < -5 or opacities.max() > 5:  # Likely logit
        opacities = torch.sigmoid(opacities)
    
    # Combine SH coefficients
    sh0 = torch.from_numpy(sh_coeffs[0]).float().to(device).unsqueeze(1)  # (N, 1, 3)
    if shN is not None:
        shN = torch.from_numpy(shN).float().to(device)  # (N, rest, 3)
        colors = torch.cat([sh0, shN], dim=1)  # (N, total_sh, 3)
        sh_degree = int(math.sqrt(colors.shape[1])) - 1
    else:
        colors = sh0
        sh_degree = 0
    
    print(f"Loaded {len(means)} Gaussians with SH degree {sh_degree}")
    
    return {
        'means': means,
        'quats': quats,
        'scales': scales,
        'opacities': opacities,
        'colors': colors,
        'sh_degree': sh_degree,
    }


def load_checkpoint(ckpt_paths: list, device: torch.device) -> dict:
    """Load one or more checkpoint files and combine them."""
    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    
    for ckpt_path in ckpt_paths:
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Handle different checkpoint formats
        if "splats" in ckpt:
            splats = ckpt["splats"]
        else:
            splats = ckpt
        
        means.append(splats["means"])
        quats.append(F.normalize(splats["quats"], p=2, dim=-1))
        scales.append(torch.exp(splats["scales"]))
        opacities.append(torch.sigmoid(splats["opacities"]))
        sh0.append(splats["sh0"])
        shN.append(splats["shN"])
    
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2])) - 1
    
    print(f"Loaded {len(means)} Gaussians with SH degree {sh_degree}")
    
    return {
        'means': means,
        'quats': quats,
        'scales': scales,
        'opacities': opacities,
        'colors': colors,
        'sh_degree': sh_degree,
    }


def load_test_data(device: torch.device, scene_grid: int = 1) -> dict:
    """Load test/demo data."""
    from gsplat._helper import load_test_data as gsplat_load_test_data
    
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = gsplat_load_test_data(device=device, scene_grid=scene_grid)
    
    print(f"Loaded test data with {len(means)} Gaussians")
    
    return {
        'means': means,
        'quats': quats,
        'scales': scales,
        'opacities': opacities,
        'colors': colors,
        'sh_degree': None,  # Test data uses direct colors
    }


def run_viewer(
    data: dict,
    device: torch.device,
    port: int = 8080,
    backend: str = "gsplat",
):
    """Run the interactive viewer server."""
    
    means = data['means']
    quats = data['quats']
    scales = data['scales']
    opacities = data['opacities']
    colors = data['colors']
    sh_degree = data['sh_degree']
    
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if backend == "gsplat":
            rasterization_fn = rasterization
        elif backend == "inria":
            from gsplat import rasterization_inria_wrapper
            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError(f"Unknown backend: {backend}")

        try:
            render_colors, render_alphas, meta = rasterization_fn(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmat[None],  # [1, 4, 4]
                K[None],  # [1, 3, 3]
                width,
                height,
                sh_degree=sh_degree,
                render_mode="RGB",
                # Speed up large-scale rendering by skipping far-away Gaussians
                radius_clip=3,
            )
            render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        except Exception as e:
            print(f"Render error: {e}")
            render_rgbs = np.zeros((height, width, 3), dtype=np.float32)
        
        return render_rgbs

    # Create and start the viewer server
    server = viser.ViserServer(port=port, verbose=False)
    viewer = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    
    print(f"\n" + "="*60)
    print(f"  Viewer running at: http://localhost:{port}")
    print(f"  Number of Gaussians: {len(means)}")
    print(f"  SH Degree: {sh_degree}")
    print(f"="*60)
    print(f"\nControls:")
    print(f"  - Left-click + drag: Rotate camera")
    print(f"  - Right-click + drag: Pan camera")
    print(f"  - Scroll: Zoom in/out")
    print(f"\nPress Ctrl+C to exit.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


def main():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a trained checkpoint
  python viewer.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt

  # View multiple checkpoint files (distributed training)
  python viewer.py --ckpt ckpt_rank0.pt ckpt_rank1.pt

  # View a PLY file
  python viewer.py --ply output.ply

  # View test data
  python viewer.py
        """
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        default=None,
        help="Path to checkpoint file(s) (.pt)",
    )
    parser.add_argument(
        "--ply",
        type=str,
        default=None,
        help="Path to PLY file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viewer server (default: 8080)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gsplat",
        choices=["gsplat", "inria"],
        help="Rasterization backend (default: gsplat)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--scene_grid",
        type=int,
        default=1,
        help="Repeat test scene into NxN grid (must be odd, default: 1)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    
    # Load data
    if args.ply is not None:
        if not os.path.exists(args.ply):
            raise FileNotFoundError(f"PLY file not found: {args.ply}")
        data = load_ply(args.ply, device)
    elif args.ckpt is not None:
        for ckpt_path in args.ckpt:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        data = load_checkpoint(args.ckpt, device)
    else:
        print("No checkpoint or PLY file specified, loading test data...")
        if args.scene_grid % 2 != 1:
            raise ValueError("scene_grid must be odd")
        data = load_test_data(device, args.scene_grid)
    
    # Run the viewer
    run_viewer(
        data=data,
        device=device,
        port=args.port,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
