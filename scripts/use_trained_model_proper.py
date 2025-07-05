#!/usr/bin/env python3
"""
Proper usage of your trained IP-Adapter model
This script shows how to use the converted checkpoint for image generation
"""

import os
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

# Add IP-Adapter to path
sys.path.append('IP-Adapter-main')
from ip_adapter.ip_adapter import IPAdapter

def test_trained_model():
    """
    Test your trained IP-Adapter model with proper checkpoint loading
    """
    print("=== Testing Your Trained IP-Adapter Model ===\n")
    
    # Check if converted checkpoint exists
    checkpoint_file = "trained_ip_adapter.safetensors"
    if not os.path.exists(checkpoint_file):
        print(f"❌ Converted checkpoint not found: {checkpoint_file}")
        print("Please run: python convert_checkpoint.py first")
        return
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"📱 Using device: {device}")
    print(f"🔢 Using dtype: {dtype}")
    
    try:
        # Load base pipeline
        print("📥 Loading Stable Diffusion pipeline...")
        base_model_path = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Load your trained IP-Adapter
        print("🔧 Loading your trained IP-Adapter...")
        ip_model = IPAdapter(
            pipe, 
            "openai/clip-vit-large-patch14",
            checkpoint_file,
            device
        )
        
        print("✅ Model loaded successfully!")
        
        # Test cases using your training images
        test_cases = [
            {
                "image": "examples/mini_dataset/images/image_0000.jpg",
                "prompt": "A majestic cat in a royal palace, oil painting",
                "output": "result_cat_royal.jpg",
                "description": "Cat reference → Royal cat"
            },
            {
                "image": "examples/mini_dataset/images/image_0001.jpg",
                "prompt": "Dramatic mountains under stormy sky, cinematic",
                "output": "result_mountains_dramatic.jpg", 
                "description": "Sunset mountains → Stormy mountains"
            },
            {
                "image": "examples/mini_dataset/images/image_0003.jpg",
                "prompt": "Futuristic sports car in cyberpunk city",
                "output": "result_car_cyberpunk.jpg",
                "description": "Red car → Cyberpunk car"
            }
        ]
        
        print(f"\n🎨 Generating images with your trained model...")
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n--- Test {i}/{len(test_cases)}: {test['description']} ---")
            
            # Load reference image
            if not os.path.exists(test["image"]):
                print(f"❌ Reference image not found: {test['image']}")
                continue
            
            reference_image = Image.open(test["image"]).convert("RGB")
            print(f"📸 Reference: {test['image']}")
            print(f"📝 Prompt: {test['prompt']}")
            print(f"🔄 Generating...")
            
            # Generate with IP-Adapter
            images = ip_model.generate(
                pil_image=reference_image,
                prompt=test["prompt"],
                negative_prompt="lowres, bad anatomy, worst quality, low quality, blurry",
                num_samples=1,
                num_inference_steps=30,
                seed=42,
                scale=1.0,  # IP-Adapter influence strength
                guidance_scale=7.5
            )
            
            # Save result
            result_image = images[0]
            result_image.save(test["output"])
            print(f"✅ Saved: {test['output']}")
        
        print(f"\n🎉 All tests completed!")
        print(f"\nGenerated images:")
        for test in test_cases:
            if os.path.exists(test["output"]):
                print(f"  ✅ {test['output']} - {test['description']}")
        
        print(f"\n📊 Model Performance Notes:")
        print(f"   • Your model was trained on only 5 images")
        print(f"   • It learned to associate image features with text prompts")
        print(f"   • Results may vary based on similarity to training data")
        print(f"   • For better results, train on a larger, more diverse dataset")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure you ran: python convert_checkpoint.py")
        print(f"  2. Check that all dependencies are installed")
        print(f"  3. Ensure you have sufficient memory/VRAM")

def generate_custom_image(reference_image_path, prompt, output_path="custom_output.jpg"):
    """
    Generate a custom image with your trained model
    """
    print(f"🎨 Custom generation:")
    print(f"   Reference: {reference_image_path}")
    print(f"   Prompt: {prompt}")
    print(f"   Output: {output_path}")
    
    # This would follow the same pattern as test_trained_model()
    # but allow for custom inputs
    
    print("💡 Use test_trained_model() to see the full implementation")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Use your trained IP-Adapter model")
    parser.add_argument("--test", action="store_true", help="Run test cases")
    parser.add_argument("--image", type=str, help="Reference image path")
    parser.add_argument("--prompt", type=str, help="Text prompt")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output path")
    
    args = parser.parse_args()
    
    if args.test or (not args.image and not args.prompt):
        test_trained_model()
    else:
        if not args.image or not args.prompt:
            print("❌ Please provide both --image and --prompt for custom generation")
            return
        generate_custom_image(args.image, args.prompt, args.output)

if __name__ == "__main__":
    main() 