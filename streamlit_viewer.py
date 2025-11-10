#!/usr/bin/env python3
"""
Image Gallery Viewer with Descriptions
A Streamlit app to view images and their corresponding descriptions from a Parquet database.
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import pyperclip
import io
import pandas as pd

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
    
    .prompt-box {
        background-color: #e7f3ff;
        color: #000000;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #b3d9ff;
        margin-bottom: 10px;
        font-size: 0.9em;
        font-style: italic;
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
        .prompt-box {
            background-color: #1a3a52;
            color: #ffffff;
            border-color: #2a5a82;
        }
    }
    
    /* Force dark mode if Streamlit is in dark theme */
    [data-testid="stAppViewContainer"][data-theme="dark"] .description-box {
        background-color: #2b2b2b;
        color: #ffffff;
        border-color: #404040;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .prompt-box {
        background-color: #1a3a52;
        color: #ffffff;
        border-color: #2a5a82;
    }
    
    /* Force light mode if Streamlit is in light theme */
    [data-testid="stAppViewContainer"][data-theme="light"] .description-box {
        background-color: #f8f9fa;
        color: #000000;
        border-color: #dee2e6;
    }
    
    [data-testid="stAppViewContainer"][data-theme="light"] .prompt-box {
        background-color: #e7f3ff;
        color: #000000;
        border-color: #b3d9ff;
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


def load_parquet_db(parquet_path: Path) -> pd.DataFrame:
    """
    Load Parquet database.
    
    Args:
        parquet_path: Path to the Parquet file
        
    Returns:
        DataFrame with image_path, prompt, and description columns
    """
    try:
        return pd.read_parquet(parquet_path)
    except Exception as e:
        st.error(f"Error loading Parquet database: {str(e)}")
        return pd.DataFrame(columns=['image_path', 'prompt', 'description'])


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


def display_image_with_description(row: pd.Series, index: int, thumbnail_size: int = 300):
    """
    Display an image with its description in a card-like layout.
    
    Args:
        row: Pandas Series with image_path, prompt, and description
        index: Index for unique key generation
        thumbnail_size: Size for the thumbnail
    """
    image_path = Path(row['image_path'])
    prompt = row['prompt']
    description = row['description']
    
    # Create a container for the image-description pair
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display thumbnail
            try:
                # Check if image exists
                if not image_path.exists():
                    st.error(f"❌ Image not found: {image_path.name}")
                    st.caption(f"📁 {image_path}")
                else:
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
                st.caption(f"📁 {image_path}")
        
        with col2:
            # Display prompt
            st.markdown("### Prompt")
            st.markdown(
                f'<div class="prompt-box">💬 {prompt}</div>',
                unsafe_allow_html=True
            )
            
            # Display description
            if description and pd.notna(description):
                st.markdown("### Description")
                
                # Display the text with proper styling
                st.markdown(
                    f'<div class="description-box">{description}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("")  # Spacing
                
                # Buttons row
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 2])
                
                with btn_col1:
                    # Copy description button
                    if st.button(f"📋 Copy Desc", key=f"copy_desc_{index}", use_container_width=True):
                        try:
                            pyperclip.copy(description)
                            st.success("✓ Copied!", icon="✅")
                        except:
                            # Fallback: show in a text input for manual copy
                            st.session_state[f'show_copy_{index}'] = True
                
                with btn_col2:
                    # Copy prompt button
                    if st.button(f"💬 Copy Prompt", key=f"copy_prompt_{index}", use_container_width=True):
                        try:
                            pyperclip.copy(prompt)
                            st.success("✓ Copied!", icon="✅")
                        except:
                            st.session_state[f'show_copy_prompt_{index}'] = True
                
                with btn_col3:
                    # Download button for the description
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
                        "Select and copy description:",
                        value=description,
                        height=100,
                        key=f"manual_copy_{index}"
                    )
                
                if st.session_state.get(f'show_copy_prompt_{index}', False):
                    st.text_area(
                        "Select and copy prompt:",
                        value=prompt,
                        height=50,
                        key=f"manual_copy_prompt_{index}"
                    )
                
                # Show character count
                st.caption(f"📝 {len(description)} characters | Full path: {image_path}")
            else:
                st.warning("⚠️ No description found in database")
        
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
        if total_pages > 1:
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


def apply_search_filter(df: pd.DataFrame, search_query: str, search_in: str) -> pd.DataFrame:
    """
    Apply search filter to the dataframe based on search query and search location.
    
    Args:
        df: DataFrame to filter
        search_query: Search query string
        search_in: Where to search - "Filename OR Description", "Description", "Filename", "Prompt", "All"
        
    Returns:
        Filtered DataFrame
    """
    if not search_query:
        return df
    
    search_lower = search_query.lower()
    
    if search_in == "Filename OR Description":
        # Extract filename from path
        df_copy = df.copy()
        df_copy['filename'] = df_copy['image_path'].apply(lambda x: Path(x).name.lower())
        
        # Search in filename OR description
        mask = (
            df_copy['filename'].str.contains(search_lower, na=False) |
            df_copy['description'].fillna('').str.lower().str.contains(search_lower, na=False)
        )
        return df[mask]
    
    elif search_in == "Description":
        mask = df['description'].fillna('').str.lower().str.contains(search_lower, na=False)
        return df[mask]
    
    elif search_in == "Filename":
        # Extract filename from path
        df_copy = df.copy()
        df_copy['filename'] = df_copy['image_path'].apply(lambda x: Path(x).name.lower())
        mask = df_copy['filename'].str.contains(search_lower, na=False)
        return df[mask]
    
    elif search_in == "Full Path":
        mask = df['image_path'].str.lower().str.contains(search_lower, na=False)
        return df[mask]
    
    elif search_in == "Prompt":
        mask = df['prompt'].str.lower().str.contains(search_lower, na=False)
        return df[mask]
    
    else:  # All
        df_copy = df.copy()
        df_copy['filename'] = df_copy['image_path'].apply(lambda x: Path(x).name.lower())
        mask = (
            df_copy['description'].fillna('').str.lower().str.contains(search_lower, na=False) |
            df_copy['image_path'].str.lower().str.contains(search_lower, na=False) |
            df_copy['prompt'].str.lower().str.contains(search_lower, na=False)
        )
        return df[mask]


def main():
    """Main function for the Streamlit app."""
    
    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Title and description
    st.title("🖼️ Image Gallery Viewer")
    st.markdown("View images and their AI-generated descriptions from Parquet database")
    
    # Sidebar for database selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Database file input
        db_path = st.text_input(
            "Parquet Database Path",
            value="./vision_ai.parquet",
            help="Enter the path to the Parquet database file"
        )
        
        # Browse button helper
        st.caption("💡 Tip: Enter the full or relative path to your Parquet database")
        
        # Validate database
        if db_path:
            parquet_path = Path(db_path)
            
            if not parquet_path.exists():
                st.error("❌ Database file does not exist")
                return
            
            if not parquet_path.is_file():
                st.error("❌ Path is not a file")
                return
            
            if parquet_path.suffix != '.parquet':
                st.warning("⚠️ File does not have .parquet extension")
            
            st.success("✓ Valid database file")
            
            # Load the Parquet database
            with st.spinner("Loading database..."):
                df = load_parquet_db(parquet_path)
            
            if df.empty:
                st.warning("⚠️ Database is empty or could not be loaded")
                return
            
            # Check for required columns
            required_columns = {'image_path', 'prompt', 'description'}
            if not required_columns.issubset(df.columns):
                st.error(f"❌ Database missing required columns. Found: {list(df.columns)}")
                return
            
            # Count images that exist
            df['exists'] = df['image_path'].apply(lambda x: Path(x).exists())
            images_exist = df['exists'].sum()
            images_missing = len(df) - images_exist
            
            # Display statistics
            st.markdown("---")
            st.markdown("### 📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Entries", len(df))
                st.metric("Images Found", images_exist)
            with col2:
                st.metric("Images Missing", images_missing)
                st.metric("Unique Prompts", df['prompt'].nunique())
            
            # Show file size and modification time
            file_size = parquet_path.stat().st_size / 1024  # KB
            if file_size > 1024:
                file_size_str = f"{file_size/1024:.2f} MB"
            else:
                file_size_str = f"{file_size:.2f} KB"
            
            st.caption(f"💾 Database size: {file_size_str}")
            
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
            
            # Filter by image existence
            existence_filter = st.radio(
                "Show",
                ["All Entries", "Images Found Only", "Images Missing Only"],
                index=0
            )
            
            if existence_filter == "Images Found Only":
                filtered_df = df[df['exists']].copy()
            elif existence_filter == "Images Missing Only":
                filtered_df = df[~df['exists']].copy()
            else:
                filtered_df = df.copy()
            
            # Filter by prompt
            unique_prompts = sorted(df['prompt'].unique())
            if len(unique_prompts) > 1:
                selected_prompts = st.multiselect(
                    "Filter by Prompt",
                    options=["All"] + unique_prompts,
                    default=["All"]
                )
                
                if "All" not in selected_prompts:
                    filtered_df = filtered_df[filtered_df['prompt'].isin(selected_prompts)]
            
            # Sorting options
            st.markdown("---")
            st.markdown("### 📑 Sorting")
            sort_option = st.selectbox(
                "Sort by",
                ["Image Name (A-Z)", "Image Name (Z-A)", "Prompt (A-Z)", "Prompt (Z-A)"]
            )
            
            # Apply sorting
            if sort_option == "Image Name (A-Z)":
                filtered_df = filtered_df.sort_values('image_path')
            elif sort_option == "Image Name (Z-A)":
                filtered_df = filtered_df.sort_values('image_path', ascending=False)
            elif sort_option == "Prompt (A-Z)":
                filtered_df = filtered_df.sort_values('prompt')
            elif sort_option == "Prompt (Z-A)":
                filtered_df = filtered_df.sort_values('prompt', ascending=False)
            
            # Search functionality
            st.markdown("---")
            st.markdown("### 🔎 Search")
            
            search_in = st.selectbox(
                "Search in",
                ["Filename OR Description", "Description", "Filename", "Full Path", "Prompt", "All"],
                index=0,
                help="Choose where to search. 'Filename OR Description' searches in both fields and returns results if found in either."
            )
            
            search_query = st.text_input("Search", "", placeholder="Enter search term...")
            
            # Show search help based on selection
            if search_in == "Filename OR Description":
                st.caption("🔍 Will match if found in filename OR description (most common use case)")
            elif search_in == "All":
                st.caption("🔍 Will search across all fields: filename, description, full path, and prompt")
            
            # Apply search filter
            filtered_df = apply_search_filter(filtered_df, search_query, search_in)
            
            # Show search results count if searching
            if search_query:
                st.info(f"Found {len(filtered_df)} matching result(s)")
            
            # Reset to page 1 if filters change
            if 'last_filter' not in st.session_state:
                st.session_state.last_filter = (existence_filter, sort_option, search_query, search_in)
            
            current_filter = (existence_filter, sort_option, search_query, search_in)
            if st.session_state.last_filter != current_filter:
                st.session_state.current_page = 1
                st.session_state.last_filter = current_filter
        else:
            st.info("👈 Enter a database path to get started")
            return
    
    # Main content area
    if filtered_df.empty:
        st.info("No entries match the current filter criteria")
        return
    
    # Display count
    st.markdown(f"### Showing {len(filtered_df)} entry/entries")
    
    # Calculate pagination
    total_pages = (len(filtered_df) - 1) // items_per_page + 1
    
    # Ensure current page is valid
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    
    current_page = st.session_state.current_page
    
    # Calculate start and end indices
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_df))
    
    # Display current page info at top
    if total_pages > 1:
        st.info(f"📄 Page {current_page} of {total_pages} | Showing items {start_idx + 1}-{end_idx} of {len(filtered_df)}")
    
    # Display images with descriptions
    page_df = filtered_df.iloc[start_idx:end_idx]
    for idx, row in page_df.iterrows():
        display_image_with_description(row, idx, thumbnail_size)
    
    # Render pagination controls at the bottom
    render_pagination(current_page, total_pages)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Image Gallery Viewer | Built with Streamlit | Powered by Parquet"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
