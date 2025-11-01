#!/usr/bin/env python3
"""
Image Analysis CLI Tool
Processes images using OpenAI-compatible vision models and saves results to text files.
"""

import os
import base64
import argparse
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_mime_type(extension: str) -> str:
    """
    Get MIME type for image extension.
    
    Args:
        extension: File extension (e.g., '.jpg', '.png')
        
    Returns:
        MIME type string
    """
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp'
    }
    return mime_types.get(extension.lower(), 'image/jpeg')


def process_image(
    image_path: Path,
    prompt: str,
    client: openai.OpenAI,
    model_name: str,
    output_dir: Path = None
) -> None:
    """
    Process a single image with the vision model and save the result.
    
    Args:
        image_path: Path to the image file
        prompt: Prompt to send to the vision model
        client: OpenAI client instance
        model_name: Name of the vision model to use
        output_dir: Optional output directory for text files
    """
    print(f"Processing: {image_path.name}")
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(str(image_path))
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
        
        # Determine output directory
        if output_dir:
            output_directory = output_dir
            output_directory.mkdir(parents=True, exist_ok=True)
        else:
            output_directory = image_path.parent
        
        # Create output text file with same name as image but .txt extension
        output_path = output_directory / f"{image_path.stem}.txt"
        
        # Write result to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        print(f"✓ Saved result to: {output_path.name}")
        
    except Exception as e:
        print(f"✗ Error processing {image_path.name}: {str(e)}")


def get_image_files(directory: Path) -> List[Path]:
    """
    Get all supported image files from a directory.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of image file paths
    """
    supported_extensions = {'.webp', '.png', '.jpg', '.jpeg'}
    image_files = []
    
    for ext in supported_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def main():
    """Main function to parse arguments and process images."""
    parser = argparse.ArgumentParser(
        description="Process images with OpenAI-compatible vision model"
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
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output directory for text files (default: same as image directory)'
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
    input_dir = Path(args.directory)
    if not input_dir.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    if not input_dir.is_dir():
        print(f"Error: '{args.directory}' is not a directory")
        return
    
    # Parse output directory if provided
    output_dir = Path(args.output) if args.output else None
    
    # Initialize OpenAI client with custom endpoint
    client = openai.OpenAI(
        base_url=api_base,
        api_key=api_key
    )
    
    print(f"Using model: {model_name}")
    print(f"API endpoint: {api_base}")
    print(f"Input directory: {input_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        print("Output directory: Same as input images")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    
    # Get all image files
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print(f"No supported image files found in '{args.directory}'")
        print("Supported formats: .webp, .png, .jpg, .jpeg")
        return
    
    print(f"Found {len(image_files)} image(s) to process\n")
    
    # Process each image
    for image_path in image_files:
        process_image(image_path, args.prompt, client, model_name, output_dir)
    
    print("-" * 60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
