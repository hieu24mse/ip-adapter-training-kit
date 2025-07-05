#!/usr/bin/env python3
"""
Train IP-Adapter using the mini dataset
This script is configured specifically for the mini dataset with 5 samples
"""

import os
import sys
import subprocess
import torch

def detect_device_and_precision():
    """Detect the best device and mixed precision setting"""
    if torch.cuda.is_available():
        device = "cuda"
        # Check if GPU supports bf16 (newer GPUs)
        if torch.cuda.get_device_capability()[0] >= 8:
            mixed_precision = "bf16"
        else:
            mixed_precision = "fp16"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        mixed_precision = "no"  # MPS doesn't support mixed precision well
    else:
        device = "cpu"
        mixed_precision = "no"
    
    return device, mixed_precision

def main():
    print("=== IP-Adapter Training with Mini Dataset ===\n")
    
    # Detect optimal settings for current hardware
    device, mixed_precision = detect_device_and_precision()
    print(f"🖥️  Detected device: {device}")
    print(f"⚡ Mixed precision: {mixed_precision}")
    print()
    
    # Configuration
    config = {
        "data_json_file": "examples/mini_dataset/data.json",
        "data_root_path": "examples/mini_dataset",
        "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        "image_encoder_path": "openai/clip-vit-large-patch14",
        "output_dir": "mini_dataset_output",
        "resolution": 512,
        "train_batch_size": 1,  # Small batch size for mini dataset
        "num_train_epochs": 10,  # Fewer epochs for testing
        "learning_rate": 1e-4,
        "save_steps": 5,  # Save every 5 steps (will save after processing all samples)
        "mixed_precision": mixed_precision,  # Auto-detected based on hardware
        "dataloader_num_workers": 0,
    }
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nDataset Info:")
    print(f"  • Dataset: {config['data_json_file']}")
    print(f"  • Images: {config['data_root_path']}/images/")
    print(f"  • 5 samples available")
    print(f"  • Output: {config['output_dir']}/")
    
    # Check if files exist
    if not os.path.exists(config['data_json_file']):
        print(f"\n❌ Error: Dataset file not found: {config['data_json_file']}")
        print("Please make sure the mini dataset exists in examples/mini_dataset/")
        return
    
    if not os.path.exists(os.path.join(config['data_root_path'], 'images')):
        print(f"\n❌ Error: Images directory not found: {config['data_root_path']}/images/")
        return
    
    # Build command
    cmd = [
        "python", "IP-Adapter-main/tutorial_train.py",
        "--pretrained_model_name_or_path", config["pretrained_model_name_or_path"],
        "--data_json_file", config["data_json_file"],
        "--data_root_path", config["data_root_path"],
        "--image_encoder_path", config["image_encoder_path"],
        "--output_dir", config["output_dir"],
        "--resolution", str(config["resolution"]),
        "--train_batch_size", str(config["train_batch_size"]),
        "--num_train_epochs", str(config["num_train_epochs"]),
        "--learning_rate", str(config["learning_rate"]),
        "--save_steps", str(config["save_steps"]),
        "--mixed_precision", config["mixed_precision"],
        "--dataloader_num_workers", str(config["dataloader_num_workers"]),
    ]
    
    print(f"\n🚀 Starting training...")
    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("  1. Download the base Stable Diffusion model (~4GB)")
    print("  2. Download the CLIP image encoder (~1GB)")
    print("  3. Train IP-Adapter on your 5 sample images")
    print("  4. Save checkpoints every 5 steps")
    print("  5. Complete training in ~10 epochs")
    print(f"  6. Using {device.upper()} acceleration" + (f" with {mixed_precision} precision" if mixed_precision != "no" else ""))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ Training completed successfully!")
        print(f"Check the output directory: {config['output_dir']}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Common issues:")
        if device == "mps":
            print("  • Mac/MPS: Training may be slower, consider using smaller batch size")
            print("  • Mac/MPS: Ensure you have sufficient unified memory (16GB+ recommended)")
        elif device == "cuda":
            print("  • Make sure you have sufficient GPU memory (at least 8GB)")
        else:
            print("  • CPU training will be very slow, consider using GPU if available")
        print("  • Check that all dependencies are installed")
        print("  • Verify internet connection for model downloads")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 