#!/usr/bin/env python3
"""
Convert Accelerate checkpoint to IP-Adapter format
This script extracts the trained IP-Adapter weights and saves them in the correct format
"""

import os
import torch
from safetensors.torch import save_file, load_file
from accelerate import Accelerator

def convert_checkpoint(input_checkpoint="mini_dataset_output/checkpoint-50", output_file="trained_ip_adapter.safetensors"):
    """
    Convert Accelerate checkpoint to IP-Adapter format
    """
    print("🔧 Converting checkpoint format...")
    print(f"📂 Input: {input_checkpoint}")
    print(f"💾 Output: {output_file}")
    
    # Load the checkpoint
    checkpoint_path = os.path.join(input_checkpoint, "model.safetensors")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Primary checkpoint not found: {checkpoint_path}")
        # Try alternative names
        alt_files = ["pytorch_model.bin", "model.bin", "adapter_model.safetensors"]
        for alt_file in alt_files:
            alt_path = os.path.join(input_checkpoint, alt_file)
            if os.path.exists(alt_path):
                print(f"🔍 Found alternative: {alt_file}")
                checkpoint_path = alt_path
                break
        else:
            print("❌ No checkpoint file found")
            return False
    
    try:
        # Load the full model state
        print("📥 Loading checkpoint...")
        if checkpoint_path.endswith('.safetensors'):
            print("   Detected SafeTensors format")
            state_dict = load_file(checkpoint_path)
        else:
            print("   Detected PyTorch format")  
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract IP-Adapter specific weights
        ip_adapter_state = {"image_proj": {}, "ip_adapter": {}}
        
        print("🔍 Extracting IP-Adapter weights...")
        
        # Look for image projection model weights
        # We need to map UNet attention processor keys to simple numeric keys
        attention_processors = []
        image_proj_keys = []
        
        for key, value in state_dict.items():
            # Handle image projection model
            if "image_proj_model" in key:
                new_key = key.replace("image_proj_model.", "").replace("module.", "")
                ip_adapter_state["image_proj"][new_key] = value
                image_proj_keys.append(new_key)
            # Handle UNet attention processors with full paths
            elif "unet." in key and any(x in key for x in ["to_k_ip", "to_v_ip"]):
                # Extract the attention processor part
                if "processor." in key:
                    processor_key = key.split("processor.")[-1]  # Get the part after "processor."
                    # Remove the "unet." prefix from the full key to get the base path
                    base_path_with_processor = key.replace("unet.", "")
                    attention_processors.append((key, processor_key, value, base_path_with_processor))
        
        # Sort attention processors by their UNet path for consistent ordering
        attention_processors.sort(key=lambda x: x[0])
        
        # Group attention processors by their base path (each attention layer has both to_k_ip and to_v_ip)
        attention_layers = {}
        for full_key, processor_key, value, base_path_with_processor in attention_processors:
            # Extract base path (everything before "processor.") - remove unet. prefix
            base_path = base_path_with_processor.split(".processor.")[0]
            if base_path not in attention_layers:
                attention_layers[base_path] = {}
            attention_layers[base_path][processor_key] = (full_key, value)
        
        # We need to match the exact order from unet.attn_processors.keys()
        # Let's create a UNet to get the correct ordering
        print("   Creating UNet to get correct attention processor order...")
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        
        # Get the attention processor names in the exact order used during training
        unet_attn_keys = list(unet.attn_processors.keys())
        
        # Filter to only cross-attention layers (attn2, not attn1)
        cross_attn_keys = [key for key in unet_attn_keys if not key.endswith("attn1.processor")]
        
        print(f"   Found {len(cross_attn_keys)} cross-attention layers in UNet")
        
        # Map our attention layers to match this order
        sorted_base_paths = []
        for key in cross_attn_keys:
            base_path = key.replace(".processor", "")
            if base_path in attention_layers:
                sorted_base_paths.append(base_path)
            else:
                print(f"   Warning: {base_path} not found in checkpoint")
        
        print(f"   Matched {len(sorted_base_paths)} layers from checkpoint")
        
        print(f"   UNet layer order:")
        for i, path in enumerate(sorted_base_paths):
            print(f"     {i}: {path}")
        
        # Map to numeric keys as expected by IP-Adapter
        # The training script creates adapter_modules = ModuleList(unet.attn_processors.values())
        # This includes ALL processors (attn1 + attn2), but only attn2 are IP-Adapter processors
        # So we need to find the index of each cross-attention processor in the full list
        
        print("   Mapping attention processors to ModuleList indices...")
        
        for i, key in enumerate(cross_attn_keys):
            base_path = key.replace(".processor", "")
            if base_path in attention_layers:
                layer_data = attention_layers[base_path]
                
                # Find the index of this processor in the full attn_processors list
                module_list_idx = unet_attn_keys.index(key)
                
                # Map both to_k_ip and to_v_ip for this layer
                if "to_k_ip.weight" in layer_data:
                    full_key, value = layer_data["to_k_ip.weight"]
                    shape = value.shape
                    simple_key = f"{module_list_idx}.to_k_ip.weight"
                    ip_adapter_state["ip_adapter"][simple_key] = value
                    print(f"   Mapped: {base_path} to_k_ip {shape} -> {simple_key}")
                
                if "to_v_ip.weight" in layer_data:
                    full_key, value = layer_data["to_v_ip.weight"]
                    shape = value.shape
                    simple_key = f"{module_list_idx}.to_v_ip.weight"
                    ip_adapter_state["ip_adapter"][simple_key] = value
                    print(f"   Mapped: {base_path} to_v_ip {shape} -> {simple_key}")
        
        print(f"   Found {len(image_proj_keys)} image projection keys")
        print(f"   Found {len(sorted_base_paths)} attention layers with {len(attention_processors)} total keys")
        
        if not ip_adapter_state["image_proj"] and not ip_adapter_state["ip_adapter"]:
            print("⚠️  No IP-Adapter weights found in standard format")
            print("🔍 Analyzing checkpoint structure...")
            
            # Print available keys for debugging
            print("Available keys in checkpoint:")
            for key in list(state_dict.keys())[:20]:  # Show first 20 keys
                print(f"   {key}")
            if len(state_dict.keys()) > 20:
                print(f"   ... and {len(state_dict.keys()) - 20} more")
            
            return False
        
        # Save in IP-Adapter format
        print("💾 Saving converted checkpoint...")
        
        # Flatten the state dict for safetensors
        flat_state_dict = {}
        for key, value in ip_adapter_state["image_proj"].items():
            flat_state_dict[f"image_proj.{key}"] = value
        for key, value in ip_adapter_state["ip_adapter"].items():
            flat_state_dict[f"ip_adapter.{key}"] = value
        
        save_file(flat_state_dict, output_file)
        
        print(f"✅ Conversion completed!")
        print(f"   Image projection weights: {len(ip_adapter_state['image_proj'])}")
        print(f"   IP-Adapter weights: {len(ip_adapter_state['ip_adapter'])}")
        print(f"   Total parameters: {len(flat_state_dict)}")
        print(f"   Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def analyze_checkpoint(checkpoint_path="mini_dataset_output/checkpoint-50"):
    """
    Analyze the structure of the checkpoint
    """
    print("🔍 Analyzing checkpoint structure...")
    
    # First, check what files are in the checkpoint directory
    if os.path.exists(checkpoint_path):
        print(f"📂 Files in {checkpoint_path}:")
        for file in os.listdir(checkpoint_path):
            file_path = os.path.join(checkpoint_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024*1024)  # MB
                print(f"   • {file} ({size:.1f}MB)")
    
    model_file = os.path.join(checkpoint_path, "model.safetensors")
    
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        # Try alternative names
        alt_files = ["pytorch_model.bin", "model.bin", "adapter_model.safetensors"]
        for alt_file in alt_files:
            alt_path = os.path.join(checkpoint_path, alt_file)
            if os.path.exists(alt_path):
                print(f"🔍 Found alternative: {alt_file}")
                model_file = alt_path
                break
        else:
            return
    
    try:
        if model_file.endswith('.safetensors'):
            print("   Loading SafeTensors format...")
            state_dict = load_file(model_file)
        else:
            print("   Loading PyTorch format...")
            state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        
        print(f"📊 Checkpoint Statistics:")
        print(f"   Total keys: {len(state_dict)}")
        
        # Categorize keys
        categories = {}
        for key in state_dict.keys():
            if "image_proj" in key.lower() or key.startswith("proj.") or key.startswith("norm."):
                category = "Image Projection"
            elif "adapter" in key.lower() or any(x in key for x in ["to_k_ip", "to_v_ip", "to_out_ip"]):
                category = "IP-Adapter"
            elif "unet" in key.lower():
                category = "UNet"
            elif "vae" in key.lower():
                category = "VAE"
            elif "text_encoder" in key.lower():
                category = "Text Encoder"
            else:
                category = "Other"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        
        print(f"\n📂 Key Categories:")
        for category, keys in categories.items():
            print(f"   {category}: {len(keys)} keys")
            if len(keys) <= 5:
                for key in keys:
                    print(f"      • {key}")
            else:
                for key in keys[:3]:
                    print(f"      • {key}")
                print(f"      ... and {len(keys)-3} more")
    
    except Exception as e:
        print(f"❌ Error analyzing checkpoint: {e}")

def main():
    print("=== IP-Adapter Checkpoint Converter ===\n")
    print("🔧 Fixed for PyTorch 2.6 compatibility (weights_only parameter)")
    print()
    
    # First, analyze the checkpoint
    analyze_checkpoint()
    
    print("\n" + "="*50 + "\n")
    
    # Try to convert
    success = convert_checkpoint()
    
    if success:
        print(f"\n🎉 Success! You can now use 'trained_ip_adapter.safetensors' with the IP-Adapter")
        print(f"\nNext steps:")
        print(f"  1. Use the converted file in your inference script")
        print(f"  2. Test with: python use_trained_model_proper.py")
    else:
        print(f"\n⚠️  Conversion failed. This might be due to:")
        print(f"   • Different checkpoint format than expected")
        print(f"   • Training script saved weights differently")
        print(f"   • Need to modify the conversion logic")
        print(f"\n💡 Try running the training with a different save format or")
        print(f"   modify this script based on the checkpoint analysis above")

if __name__ == "__main__":
    main() 