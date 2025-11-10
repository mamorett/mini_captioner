#!/usr/bin/env python3
"""
Scans a directory recursively and reports all subdirectories
that contain at least one image file (.jpg, .jpeg, .png, .webp).
"""

import os
import argparse
import sys

# --- Configuration ---
# Use a set for fast lookups. Add or remove extensions as needed.
# Extensions MUST be lowercase and start with a dot.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# ---------------------

def find_image_directories(start_path):
    """
    Walks a directory tree and finds directories containing specified image files.

    Args:
        start_path (str): The absolute or relative path to the root directory to scan.

    Returns:
        list: A sorted list of directory paths that contain at least one image.
    """
    # We use a set to store the directories. This automatically handles
    # duplicates, so a directory with 10 images is only added once.
    found_directories = set()

    # os.walk() recursively visits every directory in the tree.
    # For each directory, it yields a 3-tuple:
    # 1. dirpath: The path of the current directory.
    # 2. dirnames: A list of subdirectory names within dirpath.
    # 3. filenames: A list of file names within dirpath.
    for dirpath, _, filenames in os.walk(start_path, topdown=True):
        
        # We only care about the files in the current directory (dirpath)
        for filename in filenames:
            
            # Get the file extension and convert it to lowercase
            # os.path.splitext('photo.JPG') -> ('photo', '.JPG')
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in IMAGE_EXTENSIONS:
                # An image was found! Add the directory path to our set.
                found_directories.add(dirpath)
                
                # OPTIMIZATION:
                # Since we only care if *at least one* image exists,
                # we can stop checking other files in this *same* directory
                # and move on to the next one.
                break 
    
    # Convert the set to a list and sort it for clean, predictable output
    return sorted(list(found_directories))

def main():
    """
    Main function to parse arguments and run the scan.
    """
    # Set up a clean command-line argument parser
    parser = argparse.ArgumentParser(
        description="Find all directories containing image files (jpg, png, webp)."
    )
    
    # Add one required argument: the path to scan
    parser.add_argument(
        "scan_path",
        metavar="PATH",
        type=str,
        help="The root directory path to start scanning from."
    )

    args = parser.parse_args()
    scan_path = args.scan_path

    # --- Input Validation ---
    if not os.path.isdir(scan_path):
        print(f"Error: Path '{scan_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1) # Exit with a non-zero status code to indicate error

    try:
        # --- Run the main logic ---
        print(f"Scanning '{scan_path}' for images...")
        image_dirs = find_image_directories(scan_path)

        # --- Report the results ---
        if image_dirs:
            print(f"\n✅ Found {len(image_dirs)} directories containing images:\n")
            for dir_path in image_dirs:
                print(dir_path)
        else:
            print(f"\nℹ️ No directories with .jpg, .png, or .webp images found.")
    
    except PermissionError:
        print(f"\nError: Permission denied. Cannot scan one or more subdirectories.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()