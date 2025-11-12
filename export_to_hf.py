#!/usr/bin/env python3
"""
Hugging Face Dataset Exporter
Exports a Parquet database with image paths to Hugging Face with embedded images.
"""

import os
import base64
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import HfApi, login
from tqdm import tqdm
from PIL import Image
import io

# Load environment variables
load_dotenv()


def resize_image_to_megapixel(image: Image.Image, target_megapixels: float) -> Image.Image:
    """
    Resize an image to target megapixels maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        target_megapixels: Target size in megapixels (e.g., 1.0 for 1MP)
        
    Returns:
        Resized PIL Image object
    """
    # Get current dimensions
    original_width, original_height = image.size
    current_megapixels = (original_width * original_height) / 1_000_000
    
    # Only resize if image is larger than target
    if current_megapixels > target_megapixels:
        # Calculate the scaling factor to achieve target megapixels
        scale_factor = (target_megapixels / current_megapixels) ** 0.5
        
        # Calculate new dimensions maintaining aspect ratio
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Resize using LANCZOS (high-quality downsampling)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    return image


def image_to_base64(image_path: str, target_megapixels: Optional[float] = None) -> Optional[str]:
    """
    Convert an image file to base64 string, optionally resizing.
    
    Args:
        image_path: Path to the image file
        target_megapixels: Optional target size in megapixels
        
    Returns:
        Base64 encoded string or None if file doesn't exist
    """
    try:
        image = Image.open(image_path)
        
        # Resize if requested
        if target_megapixels:
            image = resize_image_to_megapixel(image, target_megapixels)
        
        # Convert to RGB if necessary (for PNG with transparency, WEBP, etc.)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {image_path}: {str(e)}")
        return None


def load_image_as_pil(image_path: str, target_megapixels: Optional[float] = None) -> Optional[Image.Image]:
    """
    Load an image file as PIL Image, optionally resizing.
    
    Args:
        image_path: Path to the image file
        target_megapixels: Optional target size in megapixels
        
    Returns:
        PIL Image object or None if file doesn't exist
    """
    try:
        image = Image.open(image_path)
        
        # Resize if requested
        if target_megapixels:
            original_size = image.size
            image = resize_image_to_megapixel(image, target_megapixels)
            new_size = image.size
            
            # Only log if actually resized
            if original_size != new_size:
                original_mp = (original_size[0] * original_size[1]) / 1_000_000
                new_mp = (new_size[0] * new_size[1]) / 1_000_000
                # We'll return the image without logging here to avoid spam
                # Logging will be done in the main export function
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading {image_path}: {str(e)}")
        return None


def resolve_image_path(image_path: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Resolve an image path, trying multiple strategies:
    1. Use as absolute path if it exists
    2. Try relative to provided base_dir
    3. Try relative to current working directory
    
    Args:
        image_path: Original image path from database
        base_dir: Optional base directory to resolve relative paths
        
    Returns:
        Resolved Path object or None if file cannot be found
    """
    # Try as absolute path first
    path = Path(image_path)
    if path.exists() and path.is_file():
        return path
    
    # Try relative to base_dir if provided
    if base_dir:
        relative_path = base_dir / path.name
        if relative_path.exists() and relative_path.is_file():
            return relative_path
    
    # Try relative to current working directory
    cwd_path = Path.cwd() / path.name
    if cwd_path.exists() and cwd_path.is_file():
        return cwd_path
    
    return None


def export_to_huggingface(
    parquet_path: Path,
    repo_id: str,
    base_dir: Optional[Path] = None,
    token: Optional[str] = None,
    private: bool = False,
    method: str = 'native',
    target_megapixels: Optional[float] = None
):
    """
    Export Parquet database to Hugging Face with embedded images.
    
    Args:
        parquet_path: Path to the Parquet database file
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name')
        base_dir: Base directory to resolve relative image paths
        token: Hugging Face API token (if not in environment)
        private: Whether to create a private repository
        method: 'native' (uses HF Image type) or 'base64' (stores as base64 strings)
        target_megapixels: Optional target size in megapixels for resizing
    """
    
    # Validate parquet file exists
    if not parquet_path.exists():
        print(f"Error: Parquet file '{parquet_path}' does not exist")
        return
    
    # Load the Parquet database
    print(f"Loading database from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} entries")
    
    if len(df) == 0:
        print("Error: Database is empty")
        return
    
    # Login to Hugging Face
    hf_token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token)
    else:
        print("Warning: No HF token found. You may need to run 'huggingface-cli login' first")
    
    # Prepare data with resolved image paths
    resize_msg = f" and resizing to {target_megapixels}MP" if target_megapixels else ""
    print(f"\nResolving image paths and loading images{resize_msg}...")
    processed_data = []
    missing_images = []
    resize_stats = {'resized': 0, 'unchanged': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        original_path = row['image_path']
        resolved_path = resolve_image_path(original_path, base_dir)
        
        if resolved_path is None:
            missing_images.append(original_path)
            continue
        
        if method == 'native':
            # Load as PIL Image for native HF Image support
            # First check original size for stats
            if target_megapixels:
                try:
                    with Image.open(str(resolved_path)) as temp_img:
                        original_mp = (temp_img.size[0] * temp_img.size[1]) / 1_000_000
                except:
                    original_mp = 0
            
            img = load_image_as_pil(str(resolved_path), target_megapixels)
            if img is None:
                missing_images.append(original_path)
                continue
            
            # Track resize stats
            if target_megapixels:
                new_mp = (img.size[0] * img.size[1]) / 1_000_000
                if original_mp > target_megapixels:
                    resize_stats['resized'] += 1
                else:
                    resize_stats['unchanged'] += 1
            
            entry = {
                'image': img,
                'prompt': row['prompt'],
                'description': row['description'],
                'original_path': original_path,
            }
        else:  # base64 method
            # First check original size for stats
            if target_megapixels:
                try:
                    with Image.open(str(resolved_path)) as temp_img:
                        original_mp = (temp_img.size[0] * temp_img.size[1]) / 1_000_000
                        if original_mp > target_megapixels:
                            resize_stats['resized'] += 1
                        else:
                            resize_stats['unchanged'] += 1
                except:
                    pass
            
            # Convert to base64 string
            img_base64 = image_to_base64(str(resolved_path), target_megapixels)
            if img_base64 is None:
                missing_images.append(original_path)
                continue
            
            entry = {
                'image_base64': img_base64,
                'image_filename': resolved_path.name,
                'prompt': row['prompt'],
                'description': row['description'],
                'original_path': original_path,
            }
        
        # Add timestamp columns if they exist
        if 'created_at' in row and pd.notna(row['created_at']):
            entry['created_at'] = str(row['created_at'])
        if 'modified_at' in row and pd.notna(row['modified_at']):
            entry['modified_at'] = str(row['modified_at'])
        
        processed_data.append(entry)
    
    # Report resize statistics
    if target_megapixels:
        print(f"\nResize statistics:")
        print(f"  Resized to {target_megapixels}MP: {resize_stats['resized']}")
        print(f"  Unchanged (already ≤{target_megapixels}MP): {resize_stats['unchanged']}")
    
    # Report missing images
    if missing_images:
        print(f"\n⚠ Warning: {len(missing_images)} image(s) could not be found:")
        for img_path in missing_images[:10]:  # Show first 10
            print(f"  - {img_path}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
        print()
    
    if not processed_data:
        print("Error: No valid images found to export")
        return
    
    print(f"Successfully processed {len(processed_data)} entries")
    
    # Create Hugging Face dataset
    print(f"\nCreating Hugging Face dataset using '{method}' method...")
    
    if method == 'native':
        # Use native HF Image type - automatically handles image storage
        dataset = Dataset.from_list(processed_data)
    else:
        # Store as base64 strings - more manual but always works
        dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
    
    print(f"Dataset created with {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")
    
    # Push to Hugging Face Hub
    print(f"\nPushing to Hugging Face Hub: {repo_id}")
    print(f"Repository visibility: {'Private' if private else 'Public'}")
    
    try:
        dataset.push_to_hub(
            repo_id,
            private=private,
            token=hf_token
        )
        print(f"\n✓ Successfully exported to: https://huggingface.co/datasets/{repo_id}")
        
        # Print usage instructions
        print("\n" + "="*60)
        print("Usage instructions:")
        print("="*60)
        if method == 'native':
            print(f"""
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access the data
for example in dataset['train']:
    image = example['image']  # PIL Image
    prompt = example['prompt']
    description = example['description']
    # ... use the data
""")
        else:
            print(f"""
from datasets import load_dataset
import base64
from PIL import Image
import io

# Load the dataset
dataset = load_dataset("{repo_id}")

# Decode images
for example in dataset['train']:
    # Decode base64 to PIL Image
    img_data = base64.b64decode(example['image_base64'])
    image = Image.open(io.BytesIO(img_data))
    
    prompt = example['prompt']
    description = example['description']
    # ... use the data
""")
    
    except Exception as e:
        print(f"\n✗ Error pushing to Hugging Face Hub: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your token has write permissions")
        print("3. Verify the repository name format: username/dataset-name")


def main():
    """Main function to parse arguments and export to Hugging Face."""
    
    parser = argparse.ArgumentParser(
        description="Export Parquet database to Hugging Face with embedded images"
    )
    parser.add_argument(
        '--database',
        '--db',
        type=str,
        required=True,
        help='Path to Parquet database file'
    )
    parser.add_argument(
        '--repo-id',
        '-r',
        type=str,
        required=True,
        help='Hugging Face repository ID (format: username/dataset-name)'
    )
    parser.add_argument(
        '--base-dir',
        '-b',
        type=str,
        help='Base directory to resolve relative image paths (default: try multiple locations)'
    )
    parser.add_argument(
        '--token',
        '-t',
        type=str,
        help='Hugging Face API token (default: use HF_TOKEN or HUGGINGFACE_TOKEN from .env)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create a private repository (default: public)'
    )
    parser.add_argument(
        '--method',
        '-m',
        type=str,
        choices=['native', 'base64'],
        default='native',
        help='Export method: "native" uses HF Image type (recommended), "base64" stores as strings (default: native)'
    )
    parser.add_argument(
        '--megapixels',
        '--mp',
        type=float,
        help='Resize images to target megapixels (e.g., 1.0 for 1MP), maintaining aspect ratio. Images smaller than target are not upscaled.'
    )
    
    args = parser.parse_args()
    
    # Parse paths
    parquet_path = Path(args.database).resolve()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else None
    
    # Validate repo_id format
    if '/' not in args.repo_id:
        print("Error: repo-id must be in format 'username/dataset-name'")
        return
    
    # Validate megapixels if provided
    if args.megapixels is not None and args.megapixels <= 0:
        print("Error: --megapixels must be a positive number")
        return
    
    print("="*60)
    print("Hugging Face Dataset Exporter")
    print("="*60)
    print(f"Database: {parquet_path}")
    print(f"Repository: {args.repo_id}")
    print(f"Method: {args.method}")
    if base_dir:
        print(f"Base directory: {base_dir}")
    if args.megapixels:
        print(f"Resize to: {args.megapixels}MP (maintaining aspect ratio)")
    print(f"Private: {args.private}")
    print("="*60 + "\n")
    
    # Export to Hugging Face
    export_to_huggingface(
        parquet_path=parquet_path,
        repo_id=args.repo_id,
        base_dir=base_dir,
        token=args.token,
        private=args.private,
        method=args.method,
        target_megapixels=args.megapixels
    )


if __name__ == "__main__":
    main()
