#!/usr/bin/env python3
"""
Image Gallery Viewer with Descriptions
A Streamlit app to view images and their corresponding text descriptions.
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import pyperclip
import io

# Page configuration
st.set_page_config(
    page_title="Image Gallery Viewer",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .description-box {
        background-color: #f8f9fa;
        color: #000000;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        max-height: 400px;
        overflow-y: auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .stImage {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .description-box {
            background-color: #2b2b2b;
            color: #ffffff;
            border-color: #404040;
        }
    }
    
    /* Force dark mode if Streamlit is in dark theme */
    [data-testid="stAppViewContainer"][data-theme="dark"] .description-box {
        background-color: #2b2b2b;
        color: #ffffff;
        border-color: #404040;
    }
    
    /* Force light mode if Streamlit is in light theme */
    [data-testid="stAppViewContainer"][data-theme="light"] .description-box {
        background-color: #f8f9fa;
        color: #000000;
        border-color: #dee2e6;
    }
    
    /* Navigation styling */
    .nav-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 0;
        gap: 10px;
    }
    
    .page-info {
        text-align: center;
        font-size: 1.1em;
        font-weight: 500;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 0 10px;
    }
    </style>
""", unsafe_allow_html=True)


def get_image_files(directory: Path) -> list:
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


def get_text_file_path(image_path: Path) -> Path:
    """
    Get the corresponding text file path for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Path object for the text file
    """
    return image_path.parent / f"{image_path.stem}.txt"


def read_text_file(text_path: Path) -> str:
    """
    Read content from a text file.
    
    Args:
        text_path: Path to the text file
        
    Returns:
        Content of the text file or None if not found
    """
    if text_path.exists():
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    return None


def create_thumbnail(image_path: Path, max_size: int = 300):
    """
    Create a thumbnail of the image and return it as a PIL Image.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size for the thumbnail (width or height)
        
    Returns:
        PIL Image object of the thumbnail
    """
    try:
        image = Image.open(image_path)
        
        # Make a copy to avoid modifying the original
        thumbnail = image.copy()
        
        # Calculate the thumbnail size maintaining aspect ratio
        thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return thumbnail
    except Exception as e:
        st.error(f"Error creating thumbnail: {str(e)}")
        return None


def display_image_with_description(image_path: Path, index: int, thumbnail_size: int = 300):
    """
    Display an image with its description in a card-like layout.
    
    Args:
        image_path: Path to the image file
        index: Index for unique key generation
        thumbnail_size: Size for the thumbnail
    """
    text_path = get_text_file_path(image_path)
    description = read_text_file(text_path)
    
    # Create a container for the image-description pair
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display thumbnail
            try:
                # Load original image to get dimensions
                original_image = Image.open(image_path)
                width, height = original_image.size
                file_size = image_path.stat().st_size / 1024  # KB
                
                # Create and display thumbnail
                thumbnail = create_thumbnail(image_path, thumbnail_size)
                if thumbnail:
                    st.image(thumbnail, caption=None, width=thumbnail_size)
                    st.caption(f"📁 {image_path.name}")
                    st.caption(f"📐 {width}×{height} | {file_size:.1f} KB")
                else:
                    st.error("Could not load image")
                    
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        
        with col2:
            # Display description
            if description:
                st.markdown("### Description")
                
                # Display the text with proper styling
                st.markdown(
                    f'<div class="description-box">{description}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("")  # Spacing
                
                # Buttons row
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                
                with btn_col1:
                    # Copy button
                    if st.button(f"📋 Copy", key=f"copy_{index}", use_container_width=True):
                        try:
                            pyperclip.copy(description)
                            st.success("✓ Copied!", icon="✅")
                        except:
                            # Fallback: show in a text input for manual copy
                            st.session_state[f'show_copy_{index}'] = True
                
                with btn_col2:
                    # Download button for the text
                    st.download_button(
                        label="💾 Download",
                        data=description,
                        file_name=f"{image_path.stem}.txt",
                        mime="text/plain",
                        key=f"download_{index}",
                        use_container_width=True
                    )
                
                # Show copyable text input if pyperclip fails
                if st.session_state.get(f'show_copy_{index}', False):
                    st.text_area(
                        "Select and copy:",
                        value=description,
                        height=100,
                        key=f"manual_copy_{index}"
                    )
                
                # Show file info
                text_file_size = text_path.stat().st_size if text_path.exists() else 0
                st.caption(f"📄 {text_path.name} ({text_file_size} bytes)")
            else:
                st.warning("⚠️ No description file found")
                st.caption(f"Expected: {text_path.name}")
        
        # Add a divider between entries
        st.divider()


def render_pagination(current_page: int, total_pages: int):
    """
    Render pagination controls at the bottom of the page.
    
    Args:
        current_page: Current page number (1-indexed)
        total_pages: Total number of pages
    """
    if total_pages <= 1:
        return current_page
    
    st.markdown("---")
    
    # Create columns for navigation buttons
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 2, 1, 1, 1])
    
    with col1:
        if st.button("⏮️ First", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.current_page = current_page - 1
            st.rerun()
    
    with col4:
        # Page info in the center
        st.markdown(
            f'<div class="page-info">Page {current_page} of {total_pages}</div>',
            unsafe_allow_html=True
        )
    
    with col6:
        if st.button("Next ▶️", use_container_width=True, disabled=(current_page == total_pages)):
            st.session_state.current_page = current_page + 1
            st.rerun()
    
    with col7:
        if st.button("Last ⏭️", use_container_width=True, disabled=(current_page == total_pages)):
            st.session_state.current_page = total_pages
            st.rerun()
    
    # Direct page jump
    with col4:
        st.markdown("")  # Spacing
        jump_to_page = st.selectbox(
            "Jump to page:",
            options=list(range(1, total_pages + 1)),
            index=current_page - 1,
            key="page_selector",
            label_visibility="collapsed"
        )
        if jump_to_page != current_page:
            st.session_state.current_page = jump_to_page
            st.rerun()
    
    return st.session_state.get('current_page', current_page)


def main():
    """Main function for the Streamlit app."""
    
    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Title and description
    st.title("🖼️ Image Gallery Viewer")
    st.markdown("View images and their AI-generated descriptions")
    
    # Sidebar for directory selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Directory input
        directory_path = st.text_input(
            "Directory Path",
            value="./images",
            help="Enter the path to the directory containing images and text files"
        )
        
        # Browse button helper
        st.caption("💡 Tip: Enter the full or relative path to your image directory")
        
        # Validate directory
        if directory_path:
            input_dir = Path(directory_path)
            
            if not input_dir.exists():
                st.error("❌ Directory does not exist")
                return
            
            if not input_dir.is_dir():
                st.error("❌ Path is not a directory")
                return
            
            st.success("✓ Valid directory")
            
            # Get image files
            image_files = get_image_files(input_dir)
            
            if not image_files:
                st.warning("No images found in directory")
                st.info("Supported formats: .webp, .png, .jpg, .jpeg")
                return
            
            # Count images with descriptions
            images_with_desc = sum(
                1 for img in image_files 
                if get_text_file_path(img).exists()
            )
            
            # Display statistics
            st.markdown("---")
            st.markdown("### 📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Images", len(image_files))
                st.metric("With Descriptions", images_with_desc)
            with col2:
                st.metric("Without Descriptions", len(image_files) - images_with_desc)
                completion_pct = (images_with_desc / len(image_files) * 100) if image_files else 0
                st.metric("Completion", f"{completion_pct:.0f}%")
            
            # Thumbnail size slider
            st.markdown("---")
            st.markdown("### 🖼️ Display")
            thumbnail_size = st.slider(
                "Thumbnail Size",
                min_value=150,
                max_value=500,
                value=300,
                step=50,
                help="Adjust the maximum size of image thumbnails"
            )
            
            # Items per page
            items_per_page = st.selectbox(
                "Items per page",
                [5, 10, 20, 50, 100],
                index=1,
                key="items_per_page"
            )
            
            # Filter options
            st.markdown("---")
            st.markdown("### 🔍 Filters")
            
            filter_option = st.radio(
                "Show",
                ["All Images", "With Descriptions Only", "Without Descriptions Only"],
                index=0
            )
            
            # Sorting options
            sort_option = st.selectbox(
                "Sort by",
                ["Name (A-Z)", "Name (Z-A)", "Date Modified (Newest)", "Date Modified (Oldest)"]
            )
            
            # Apply filters
            if filter_option == "With Descriptions Only":
                filtered_images = [
                    img for img in image_files 
                    if get_text_file_path(img).exists()
                ]
            elif filter_option == "Without Descriptions Only":
                filtered_images = [
                    img for img in image_files 
                    if not get_text_file_path(img).exists()
                ]
            else:
                filtered_images = image_files
            
            # Apply sorting
            if sort_option == "Name (A-Z)":
                filtered_images = sorted(filtered_images, key=lambda x: x.name)
            elif sort_option == "Name (Z-A)":
                filtered_images = sorted(filtered_images, key=lambda x: x.name, reverse=True)
            elif sort_option == "Date Modified (Newest)":
                filtered_images = sorted(filtered_images, key=lambda x: x.stat().st_mtime, reverse=True)
            elif sort_option == "Date Modified (Oldest)":
                filtered_images = sorted(filtered_images, key=lambda x: x.stat().st_mtime)
            
            # Search functionality
            st.markdown("---")
            search_query = st.text_input("🔎 Search in descriptions", "")
            
            if search_query:
                filtered_images = [
                    img for img in filtered_images
                    if read_text_file(get_text_file_path(img)) and 
                    search_query.lower() in read_text_file(get_text_file_path(img)).lower()
                ]
            
            # Reset to page 1 if filters change
            if 'last_filter' not in st.session_state:
                st.session_state.last_filter = (filter_option, sort_option, search_query)
            
            current_filter = (filter_option, sort_option, search_query)
            if st.session_state.last_filter != current_filter:
                st.session_state.current_page = 1
                st.session_state.last_filter = current_filter
        else:
            st.info("👈 Enter a directory path to get started")
            return
    
    # Main content area
    if not filtered_images:
        st.info("No images match the current filter criteria")
        return
    
    # Display count
    st.markdown(f"### Showing {len(filtered_images)} image(s)")
    
    # Calculate pagination
    total_pages = (len(filtered_images) - 1) // items_per_page + 1
    
    # Ensure current page is valid
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    
    current_page = st.session_state.current_page
    
    # Calculate start and end indices
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_images))
    
    # Display current page info at top
    if total_pages > 1:
        st.info(f"📄 Page {current_page} of {total_pages} | Showing items {start_idx + 1}-{end_idx} of {len(filtered_images)}")
    
    # Display images with descriptions
    for idx, image_path in enumerate(filtered_images[start_idx:end_idx], start=start_idx):
        display_image_with_description(image_path, idx, thumbnail_size)
    
    # Render pagination controls at the bottom
    render_pagination(current_page, total_pages)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Image Gallery Viewer | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
