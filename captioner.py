#!/usr/bin/env python3
"""
Image Analysis CLI Tool
Processes images using OpenAI-compatible vision models and saves results to a Parquet database.
"""

import os
import base64
import argparse
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from PIL import Image
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import signal
import sys

# Load environment variables from .env file
load_dotenv()

# Global variables for signal handling
new_entries = []
db_df = None
parquet_path = None
override_mode = False
images_to_process = []


def resize_image_to_megapixel(image_path: str, target_megapixels: float = 1.0) -> str:
    """
    Resize an image to target megapixels in memory and encode to base64.
    Maintains aspect ratio and does NOT modify the original file.
    
    Args:
        image_path: Path to the image file
        target_megapixels: Target size in megapixels (default: 1.0)
        
    Returns:
        Base64 encoded string of the resized image
    """
    # Open the image
    image = Image.open(image_path)
    
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


def get_image_mime_type(extension: str) -> str:
    """
    Get MIME type for image extension.
    Since we're converting to JPEG for transmission, always return JPEG mime type.
    
    Args:
        extension: File extension (e.g., '.jpg', '.png')
        
    Returns:
        MIME type string
    """
    return 'image/jpeg'


def load_parquet_db(parquet_path: Path) -> pd.DataFrame:
    """
    Load existing Parquet database or create empty DataFrame with correct schema.
    
    Args:
        parquet_path: Path to the Parquet file
        
    Returns:
        DataFrame with image_path, prompt, and description columns
    """
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    else:
        # Create empty DataFrame with correct schema
        return pd.DataFrame(columns=['image_path', 'prompt', 'description'])


def save_parquet_db(df: pd.DataFrame, parquet_path: Path):
    """
    Save DataFrame to Parquet file.
    
    Args:
        df: DataFrame to save
        parquet_path: Path to save the Parquet file
    """
    # Ensure parent directory exists
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet
    df.to_parquet(parquet_path, index=False, engine='pyarrow')


def save_progress():
    """
    Save current progress to the Parquet database.
    Called when Ctrl-C is pressed or at the end of processing.
    """
    global new_entries, db_df, parquet_path, override_mode
    
    if new_entries and db_df is not None and parquet_path is not None:
        try:
            new_df = pd.DataFrame(new_entries)
            
            if override_mode:
                # Remove old entries for processed images
                processed_paths = [entry['image_path'] for entry in new_entries]
                db_df_filtered = db_df[~db_df['image_path'].isin(processed_paths)]
            else:
                db_df_filtered = db_df
            
            # Append new entries
            updated_df = pd.concat([db_df_filtered, new_df], ignore_index=True)
            
            # Save updated database
            save_parquet_db(updated_df, parquet_path)
            return len(new_entries), len(updated_df)
        except Exception as e:
            print(f"\n✗ Error saving database: {str(e)}", file=sys.stderr)
            return 0, 0
    return 0, 0


def signal_handler(sig, frame):
    """
    Handle Ctrl-C (SIGINT) gracefully by saving progress before exit.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    print("\n\n⚠ Interrupt received (Ctrl-C). Saving progress...")
    
    saved_count, total_count = save_progress()
    
    if saved_count > 0:
        print(f"✓ Progress saved: {saved_count} new entries added to database")
        print(f"  Total entries in database: {total_count}")
        print(f"  Database: {parquet_path}")
    else:
        print("⊘ No progress to save")
    
    print("\nExiting...")
    sys.exit(0)


def should_process_image(image_path: Path, df: pd.DataFrame, override: bool) -> bool:
    """
    Check if an image should be processed based on existence in Parquet database.
    
    Args:
        image_path: Full path to the image file
        df: DataFrame containing existing entries
        override: Whether to override existing entries
        
    Returns:
        True if image should be processed, False otherwise
    """
    if override:
        return True
    
    # Check if image_path already exists in database
    image_path_str = str(image_path)
    return image_path_str not in df['image_path'].values


def process_image(
    image_path: Path,
    prompt: str,
    client: openai.OpenAI,
    model_name: str,
    pbar: tqdm = None
) -> tuple[bool, str, str]:
    """
    Process a single image with the vision model.
    
    Args:
        image_path: Full path to the image file
        prompt: Prompt to send to the vision model
        client: OpenAI client instance
        model_name: Name of the vision model to use
        pbar: Progress bar instance
        
    Returns:
        Tuple of (success: bool, message: str, description: str)
    """
    try:
        # Resize image to 1 megapixel and encode to base64 (in memory only)
        base64_image = resize_image_to_megapixel(str(image_path), target_megapixels=1.0)
        mime_type = get_image_mime_type(image_path.suffix)
        
        # Create the message with image
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        # Extract the response text
        result_text = response.choices[0].message.content
        
        if pbar:
            pbar.set_postfix_str(f"✓ {image_path.name}")
        
        return True, "Success", result_text
        
    except Exception as e:
        if pbar:
            pbar.set_postfix_str(f"✗ {image_path.name}")
        return False, f"Error: {str(e)}", ""


def get_image_files(directory: Path) -> List[Path]:
    """
    Get all supported image files from a directory with full paths.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of absolute image file paths
    """
    supported_extensions = {'.webp', '.png', '.jpg', '.jpeg'}
    image_files = []
    
    for ext in supported_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    # Convert to absolute paths
    return sorted([img.resolve() for img in image_files])


def main():
    """Main function to parse arguments and process images."""
    global new_entries, db_df, parquet_path, override_mode, images_to_process
    
    # Register signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Process images with OpenAI-compatible vision model and save to Parquet database"
    )
    parser.add_argument(
        '--directory',
        '-d',
        type=str,
        required=True,
        help='Directory containing images to process'
    )
    parser.add_argument(
        '--prompt',
        '-p',
        type=str,
        required=True,
        help='Prompt to send to the vision model for each image'
    )
    parser.add_argument(
        '--database',
        '--db',
        type=str,
        default='vision_ai.parquet',
        help='Path to Parquet database file (default: vision_ai.parquet in current directory)'
    )
    parser.add_argument(
        '--override',
        action='store_true',
        help='Override existing entries in database for images being processed (default: skip existing)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    api_base = os.getenv('OPENAI_API_BASE')
    api_key = os.getenv('OPENAI_API_KEY')
    model_name = os.getenv('OPENAI_MODEL_NAME')
    
    # Validate environment variables
    if not api_base:
        print("Error: OPENAI_API_BASE not found in .env file")
        return
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    if not model_name:
        print("Error: OPENAI_MODEL_NAME not found in .env file")
        return
    
    # Validate directory
    input_dir = Path(args.directory).resolve()
    if not input_dir.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    if not input_dir.is_dir():
        print(f"Error: '{args.directory}' is not a directory")
        return
    
    # Parse database path and set global variable
    parquet_path = Path(args.database).resolve()
    override_mode = args.override
    
    # Load existing Parquet database
    db_df = load_parquet_db(parquet_path)
    
    # Initialize OpenAI client with custom endpoint
    client = openai.OpenAI(
        base_url=api_base,
        api_key=api_key
    )
    
    print(f"Using model: {model_name}")
    print(f"API endpoint: {api_base}")
    print(f"Input directory: {input_dir}")
    print(f"Parquet database: {parquet_path}")
    print(f"Existing entries in database: {len(db_df)}")
    print(f"Override existing: {args.override}")
    print(f"Image processing: Resizing to 1 megapixel (in memory, maintaining aspect ratio)")
    print(f"Prompt: {args.prompt}")
    print(f"\n💡 Tip: Press Ctrl-C anytime to save progress and exit gracefully")
    print("-" * 60)
    
    # Get all image files with full paths
    all_image_files = get_image_files(input_dir)
    
    if not all_image_files:
        print(f"No supported image files found in '{args.directory}'")
        print("Supported formats: .webp, .png, .jpg, .jpeg")
        return
    
    print(f"Found {len(all_image_files)} image(s) total")
    
    # Filter images based on idempotency
    images_to_process = [
        img for img in all_image_files 
        if should_process_image(img, db_df, args.override)
    ]
    
    skipped_count = len(all_image_files) - len(images_to_process)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} image(s) already in database")
    
    if not images_to_process:
        print("\nNo images to process. All images already exist in database.")
        print("Use --override to reprocess all images.")
        return
    
    print(f"Processing {len(images_to_process)} image(s)\n")
    
    # Process each image with progress bar
    success_count = 0
    error_count = 0
    new_entries = []
    
    with tqdm(
        total=len(images_to_process),
        desc="Processing images",
        unit="img",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    ) as pbar:
        for image_path in images_to_process:
            success, message, description = process_image(
                image_path, 
                args.prompt, 
                client, 
                model_name,
                pbar
            )
            
            if success:
                success_count += 1
                # Add new entry
                new_entries.append({
                    'image_path': str(image_path),
                    'prompt': args.prompt,
                    'description': description
                })
            else:
                error_count += 1
                tqdm.write(f"✗ {image_path.name}: {message}")
            
            pbar.update(1)
    
    # Save progress at the end
    print("\nSaving results to database...")
    saved_count, total_count = save_progress()
    
    if saved_count > 0:
        print(f"✓ Database updated: {parquet_path}")
        print(f"  New entries added: {saved_count}")
        print(f"  Total entries in database: {total_count}")
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"✓ Successfully processed: {success_count}")
    if error_count > 0:
        print(f"✗ Errors: {error_count}")
    if skipped_count > 0:
        print(f"⊘ Skipped (already in database): {skipped_count}")


if __name__ == "__main__":
    main()
