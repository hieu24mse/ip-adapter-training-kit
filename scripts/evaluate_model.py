#!/usr/bin/env python3
"""
IP-Adapter Model Evaluation Script
Provides comprehensive evaluation metrics for your trained model
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json
from datetime import datetime

# Add IP-Adapter to path
sys.path.append('IP-Adapter-main')
from ip_adapter.ip_adapter import IPAdapter
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

class ModelEvaluator:
    def __init__(self, checkpoint_file="trained_ip_adapter.safetensors"):
        self.checkpoint_file = checkpoint_file
        self.device = self._detect_device()
        self.results = {}
        
    def _detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def setup_models(self):
        print("🔧 Setting up evaluation models...")
        
        # Load CLIP for similarity metrics
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model.to(self.device)
        
        # Load IP-Adapter
        if not os.path.exists(self.checkpoint_file):
            print(f"❌ Checkpoint not found: {self.checkpoint_file}")
            return False
            
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        self.ip_model = IPAdapter(
            pipe, 
            "openai/clip-vit-large-patch14",
            self.checkpoint_file,
            self.device
        )
        
        print("✅ Models loaded successfully!")
        return True
    
    def calculate_clip_similarity(self, image1, image2):
        """Calculate CLIP similarity between images"""
        inputs = self.clip_processor(images=[image1, image2], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
        
        similarity = torch.cosine_similarity(image_features[0:1], image_features[1:2])
        return similarity.item()
    
    def calculate_text_similarity(self, image, text):
        """Calculate CLIP similarity between image and text"""
        inputs = self.clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            similarity = outputs.logits_per_image[0, 0]
        
        return similarity.item()
    
    def evaluate_test_cases(self):
        """Run standard evaluation test cases"""
        print("\n📊 Running Evaluation Test Cases...")
        
        test_cases = [
            {
                'reference': 'examples/mini_dataset/images/image_0000.jpg',
                'prompt': 'A majestic cat in a royal palace, oil painting',
                'description': 'Cat → Royal Palace'
            },
            {
                'reference': 'examples/mini_dataset/images/image_0001.jpg',
                'prompt': 'Dramatic mountains under stormy sky, cinematic',
                'description': 'Mountains → Stormy Drama'
            },
            {
                'reference': 'examples/mini_dataset/images/image_0003.jpg',
                'prompt': 'Futuristic sports car in cyberpunk city',
                'description': 'Car → Cyberpunk'
            }
        ]
        
        fidelity_scores = []
        adherence_scores = []
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"   Test {i}: {test_case['description']}")
            
            # Load reference
            ref_image = Image.open(test_case['reference']).convert("RGB")
            
            # Generate image
            generated_images = self.ip_model.generate(
                pil_image=ref_image,
                prompt=test_case['prompt'],
                negative_prompt="lowres, bad anatomy, worst quality, low quality",
                num_samples=1,
                num_inference_steps=30,
                seed=42,
                scale=1.0,
                guidance_scale=7.5
            )
            
            generated_image = generated_images[0]
            
            # Calculate metrics
            ref_similarity = self.calculate_clip_similarity(ref_image, generated_image)
            text_similarity = self.calculate_text_similarity(generated_image, test_case['prompt'])
            
            fidelity_scores.append(ref_similarity)
            adherence_scores.append(text_similarity)
            
            # Save result
            output_path = f"eval_result_{i}.jpg"
            generated_image.save(output_path)
            
            result = {
                'description': test_case['description'],
                'reference_fidelity': ref_similarity,
                'prompt_adherence': text_similarity,
                'output_path': output_path
            }
            results.append(result)
            
            print(f"      Fidelity: {ref_similarity:.3f}, Adherence: {text_similarity:.3f}")
        
        return {
            'test_results': results,
            'average_fidelity': np.mean(fidelity_scores),
            'average_adherence': np.mean(adherence_scores),
            'fidelity_scores': fidelity_scores,
            'adherence_scores': adherence_scores
        }
    
    def test_scale_sensitivity(self):
        """Test different IP-Adapter scale values"""
        print("\n⚖️ Testing Scale Sensitivity...")
        
        scales = [0.5, 1.0, 1.5, 2.0]
        ref_image = Image.open('examples/mini_dataset/images/image_0000.jpg').convert("RGB")
        prompt = "A beautiful cat portrait"
        
        scale_results = []
        
        for scale in scales:
            print(f"   Testing scale: {scale}")
            
            generated_images = self.ip_model.generate(
                pil_image=ref_image,
                prompt=prompt,
                num_samples=1,
                num_inference_steps=30,
                seed=42,
                scale=scale,
                guidance_scale=7.5
            )
            
            generated_image = generated_images[0]
            
            # Calculate metrics
            ref_sim = self.calculate_clip_similarity(ref_image, generated_image)
            text_sim = self.calculate_text_similarity(generated_image, prompt)
            
            # Save image
            output_path = f"eval_scale_{scale}.jpg"
            generated_image.save(output_path)
            
            result = {
                'scale': scale,
                'reference_similarity': ref_sim,
                'text_similarity': text_sim,
                'output_path': output_path
            }
            scale_results.append(result)
            
            print(f"      Ref: {ref_sim:.3f}, Text: {text_sim:.3f}")
        
        return {'scale_results': scale_results}
    
    def generate_report(self):
        """Generate evaluation report"""
        print("\n📋 Generating Evaluation Report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': self.checkpoint_file,
            'device': self.device,
            'results': self.results,
            'summary': self._create_summary()
        }
        
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _create_summary(self):
        """Create evaluation summary"""
        summary = {}
        
        if 'test_cases' in self.results:
            tc = self.results['test_cases']
            summary['reference_fidelity'] = {
                'score': tc['average_fidelity'],
                'interpretation': self._interpret_fidelity(tc['average_fidelity'])
            }
            summary['prompt_adherence'] = {
                'score': tc['average_adherence'],
                'interpretation': self._interpret_adherence(tc['average_adherence'])
            }
        
        return summary
    
    def _interpret_fidelity(self, score):
        if score > 0.7: return "Excellent reference preservation"
        elif score > 0.5: return "Good reference preservation"
        elif score > 0.3: return "Fair reference preservation"
        else: return "Poor reference preservation"
    
    def _interpret_adherence(self, score):
        if score > 25: return "Excellent prompt following"
        elif score > 20: return "Good prompt following"
        elif score > 15: return "Fair prompt following"
        else: return "Poor prompt following"
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📈 EVALUATION SUMMARY")
        print("="*60)
        
        if 'test_cases' in self.results:
            tc = self.results['test_cases']
            print(f"Reference Fidelity: {tc['average_fidelity']:.3f}")
            print(f"  → {self._interpret_fidelity(tc['average_fidelity'])}")
            print(f"Prompt Adherence: {tc['average_adherence']:.3f}")
            print(f"  → {self._interpret_adherence(tc['average_adherence'])}")
        
        print("\n📁 Generated Files:")
        print("  • evaluation_report.json - Detailed results")
        print("  • eval_result_*.jpg - Test case outputs")
        print("  • eval_scale_*.jpg - Scale sensitivity tests")

def main():
    print("=== IP-Adapter Model Evaluation ===\n")
    
    evaluator = ModelEvaluator()
    
    if not evaluator.setup_models():
        return
    
    # Run evaluations
    evaluator.results['test_cases'] = evaluator.evaluate_test_cases()
    evaluator.results['scale_sensitivity'] = evaluator.test_scale_sensitivity()
    
    # Generate report and summary
    evaluator.generate_report()
    evaluator.print_summary()
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main() 