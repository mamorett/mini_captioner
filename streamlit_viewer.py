#!/usr/bin/env python3
"""
Image Gallery Viewer with Descriptions
A Streamlit app to view images and their corresponding descriptions from a Parquet database.

Usage:
    streamlit run viewer.py
    streamlit run viewer.py -- --database /path/to/database.parquet
    streamlit run viewer.py -- --db ./vision_ai.parquet
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import pyperclip
import io
import pandas as pd
import sys
import argparse
from datetime import datetime

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
    
    .timestamp-box {
        background-color: #f0f0f0;
        color: #000000;
        padding: 8px;
        border-radius: 5px;
        border: 1px solid #d0d0d0;
        margin-top: 10px;
        font-size: 0.85em;
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
        .timestamp-box {
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
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .prompt-box {
        background-color: #1a3a52;
        color: #ffffff;
        border-color: #2a5a82;
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .timestamp-box {
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
    
    [data-testid="stAppViewContainer"][data-theme="light"] .prompt-box {
        background-color: #e7f3ff;
        color: #000000;
        border-color: #b3d9ff;
    }
    
    [data-testid="stAppViewContainer"][data-theme="light"] .timestamp-box {
        background-color: #f0f0f0;
        color: #000000;
        border-color: #d0d0d0;
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


def parse_cli_args():
    """
    Parse command line arguments.
    Streamlit passes arguments after '--' to the script.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Image Gallery Viewer with Parquet database support"
    )
    parser.add_argument(
        '--database',
        '--db',
        dest='database',
        type=str,
        default=None,
        help='Path to Parquet database file'
    )
    
    args_to_parse = []
    
    if 'streamlit' in sys.argv[0].lower() or any('streamlit' in arg.lower() for arg in sys.argv):
        script_found = False
        for arg in sys.argv:
            if script_found:
                args_to_parse.append(arg)
            elif arg.endswith('.py'):
                script_found = True
    else:
        args_to_parse = sys.argv[1:]
    
    if not args_to_parse:
        return parser.parse_args([])
    
    return parser.parse_args(args_to_parse)


def load_parquet_db(parquet_path: Path) -> pd.DataFrame:
    """Load Parquet database."""
    try:
        df = pd.read_parquet(parquet_path)
        
        # Ensure datetime columns are properly typed
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        if 'modified_at' in df.columns:
            df['modified_at'] = pd.to_datetime(df['modified_at'])
        
        return df
    except Exception as e:
        st.error(f"Error loading Parquet database: {str(e)}")
        return pd.DataFrame(columns=['image_path', 'prompt', 'description', 'created_at', 'modified_at'])


def save_parquet_db(df: pd.DataFrame, parquet_path: Path) -> bool:
    """Save DataFrame to Parquet file."""
    try:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        return True
    except Exception as e:
        st.error(f"Error saving Parquet database: {str(e)}")
        return False


def create_thumbnail(image_path: Path, max_size: int = 300):
    """Create a thumbnail of the image."""
    try:
        image = Image.open(image_path)
        thumbnail = image.copy()
        thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return thumbnail
    except Exception as e:
        st.error(f"Error creating thumbnail: {str(e)}")
        return None


def format_datetime(dt) -> str:
    """Format datetime for display."""
    if pd.isna(dt):
        return "N/A"
    
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "N/A"


def display_image_with_description(row: pd.Series, index: int, thumbnail_size: int = 300, df_key: str = "main_df"):
    """Display an image with its description."""
    image_path = Path(row['image_path'])
    prompt = row['prompt']
    description = row['description']
    created_at = row.get('created_at', None)
    modified_at = row.get('modified_at', None)
    
    edit_key = f"edit_mode_{index}"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = False
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            try:
                if not image_path.exists():
                    st.error(f"❌ Image not found: {image_path.name}")
                    st.caption(f"📁 {image_path}")
                else:
                    original_image = Image.open(image_path)
                    width, height = original_image.size
                    file_size = image_path.stat().st_size / 1024
                    
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
            st.markdown("### Prompt")
            st.markdown(
                f'<div class="prompt-box">💬 {prompt}</div>',
                unsafe_allow_html=True
            )
            
            # Display timestamps if available
            if created_at is not None or modified_at is not None:
                timestamp_text = ""
                if created_at is not None and not pd.isna(created_at):
                    timestamp_text += f"📅 Created: {format_datetime(created_at)}"
                if modified_at is not None and not pd.isna(modified_at):
                    if timestamp_text:
                        timestamp_text += " | "
                    timestamp_text += f"✏️ Modified: {format_datetime(modified_at)}"
                
                if timestamp_text:
                    st.markdown(
                        f'<div class="timestamp-box">{timestamp_text}</div>',
                        unsafe_allow_html=True
                    )
            
            if description and pd.notna(description):
                st.markdown("### Description")
                
                if st.session_state[edit_key]:
                    edited_description = st.text_area(
                        "Edit description:",
                        value=description,
                        height=300,
                        key=f"edit_textarea_{index}",
                        label_visibility="collapsed"
                    )
                    
                    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
                    
                    with btn_col1:
                        if st.button("💾 Save", key=f"save_{index}", use_container_width=True, type="primary"):
                            df = st.session_state[df_key]
                            mask = df['image_path'] == str(image_path)
                            df.loc[mask, 'description'] = edited_description
                            
                            # Update modified_at timestamp
                            if 'modified_at' in df.columns:
                                df.loc[mask, 'modified_at'] = pd.Timestamp.now()
                            
                            st.session_state[df_key] = df
                            
                            if save_parquet_db(df, st.session_state.parquet_path):
                                st.session_state[edit_key] = False
                                st.success("✓ Description saved!", icon="✅")
                                st.rerun()
                            else:
                                st.error("Failed to save changes")
                    
                    with btn_col2:
                        if st.button("❌ Cancel", key=f"cancel_{index}", use_container_width=True):
                            st.session_state[edit_key] = False
                            st.rerun()
                    
                    st.caption(f"📝 {len(edited_description)} characters")
                
                else:
                    st.markdown(
                        f'<div class="description-box">{description}</div>',
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("")
                    
                    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1, 1, 1, 1, 1])
                    
                    with btn_col1:
                        if st.button(f"✏️ Edit", key=f"edit_{index}", use_container_width=True):
                            st.session_state[edit_key] = True
                            st.rerun()
                    
                    with btn_col2:
                        if st.button(f"📋 Copy Desc", key=f"copy_desc_{index}", use_container_width=True):
                            try:
                                pyperclip.copy(description)
                                st.toast("✓ Description copied!", icon="✅")
                            except:
                                st.session_state[f'show_copy_{index}'] = True
                    
                    with btn_col3:
                        if st.button(f"💬 Copy Prompt", key=f"copy_prompt_{index}", use_container_width=True):
                            try:
                                pyperclip.copy(prompt)
                                st.toast("✓ Prompt copied!", icon="✅")
                            except:
                                st.session_state[f'show_copy_prompt_{index}'] = True
                    
                    with btn_col4:
                        if st.button(f"📁 Copy Path", key=f"copy_path_{index}", use_container_width=True):
                            try:
                                pyperclip.copy(str(image_path))
                                st.toast("✓ Path copied!", icon="✅")
                            except:
                                st.session_state[f'show_copy_path_{index}'] = True
                    
                    with btn_col5:
                        st.download_button(
                            label="💾 Download",
                            data=description,
                            file_name=f"{image_path.stem}.txt",
                            mime="text/plain",
                            key=f"download_{index}",
                            use_container_width=True
                        )
                    
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
                    
                    if st.session_state.get(f'show_copy_path_{index}', False):
                        st.text_area(
                            "Select and copy path:",
                            value=str(image_path),
                            height=50,
                            key=f"manual_copy_path_{index}"
                        )

                    
                    st.caption(f"📝 {len(description)} characters | Full path: {image_path}")
            else:
                st.warning("⚠️ No description found in database")
        
        st.divider()


def render_pagination(current_page: int, total_pages: int):
    """Render pagination controls with slider."""
    if total_pages <= 1:
        return current_page
    
    st.markdown("---")
    
    # Navigation buttons row
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("⏮️ First", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.current_page = 1
            st.rerun()
    
    with col2:
        if st.button("◀️ Previous", use_container_width=True, disabled=(current_page == 1)):
            st.session_state.current_page = current_page - 1
            st.rerun()
    
    with col3:
        if st.button("Next ▶️", use_container_width=True, disabled=(current_page == total_pages)):
            st.session_state.current_page = current_page + 1
            st.rerun()
    
    with col4:
        if st.button("Last ⏭️", use_container_width=True, disabled=(current_page == total_pages)):
            st.session_state.current_page = total_pages
            st.rerun()
    
    # Slider for page navigation - without a static key
    st.markdown("")  # Small spacing
    new_page = st.slider(
        f"📄 Page {current_page} of {total_pages}",
        min_value=1,
        max_value=total_pages,
        value=current_page,
        step=1
    )
    
    if new_page != current_page:
        st.session_state.current_page = new_page
        st.rerun()
    
    return st.session_state.get('current_page', current_page)


def apply_search_filter(df: pd.DataFrame, search_query: str, search_in: str) -> pd.DataFrame:
    """Apply search filter to the dataframe."""
    if not search_query:
        return df
    
    search_lower = search_query.lower()
    
    if search_in == "Filename OR Description":
        df_copy = df.copy()
        df_copy['filename'] = df_copy['image_path'].apply(lambda x: Path(x).name.lower())
        mask = (
            df_copy['filename'].str.contains(search_lower, na=False) |
            df_copy['description'].fillna('').str.lower().str.contains(search_lower, na=False)
        )
        return df[mask]
    
    elif search_in == "Description":
        mask = df['description'].fillna('').str.lower().str.contains(search_lower, na=False)
        return df[mask]
    
    elif search_in == "Filename":
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


def apply_sorting(df: pd.DataFrame, sort_option: str) -> pd.DataFrame:
    """Apply sorting to the dataframe."""
    if sort_option == "Image Name (A-Z)":
        return df.sort_values('image_path')
    elif sort_option == "Image Name (Z-A)":
        return df.sort_values('image_path', ascending=False)
    elif sort_option == "Prompt (A-Z)":
        return df.sort_values('prompt')
    elif sort_option == "Prompt (Z-A)":
        return df.sort_values('prompt', ascending=False)
    elif sort_option == "Created Date (Newest First)":
        if 'created_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('created_at', ascending=False, na_position='last')
        else:
            st.warning("⚠️ 'created_at' column not found in database")
            return df
    elif sort_option == "Created Date (Oldest First)":
        if 'created_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('created_at', ascending=True, na_position='last')
        else:
            st.warning("⚠️ 'created_at' column not found in database")
            return df
    elif sort_option == "Modified Date (Newest First)":
        if 'modified_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('modified_at', ascending=False, na_position='last')
        else:
            st.warning("⚠️ 'modified_at' column not found in database")
            return df
    elif sort_option == "Modified Date (Oldest First)":
        if 'modified_at' in df.columns:
            # Put NaT values at the end
            return df.sort_values('modified_at', ascending=True, na_position='last')
        else:
            st.warning("⚠️ 'modified_at' column not found in database")
            return df
    else:
        return df


def main():
    """Main function for the Streamlit app."""
    
    cli_args = parse_cli_args()
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    if 'cli_database_path' not in st.session_state:
        st.session_state.cli_database_path = cli_args.database
    
    st.title("🖼️ Image Gallery Viewer")
    st.markdown("View images and their AI-generated descriptions from Parquet database")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if st.session_state.cli_database_path:
            default_db_path = st.session_state.cli_database_path
            st.info(f"📌 Database set via CLI: {Path(default_db_path).name}")
        else:
            default_db_path = "./vision_ai.parquet"
        
        db_path = st.text_input(
            "Parquet Database Path",
            value=default_db_path,
            help="Enter the path to the Parquet database file"
        )
        
        st.caption("💡 Tip: Enter the full or relative path to your Parquet database")
        if st.session_state.cli_database_path:
            st.caption("🔧 Database path was provided via CLI argument")
        
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
            
            st.session_state.parquet_path = parquet_path
            
            with st.spinner("Loading database..."):
                df = load_parquet_db(parquet_path)
            
            if df.empty:
                st.warning("⚠️ Database is empty or could not be loaded")
                return
            
            required_columns = {'image_path', 'prompt', 'description'}
            if not required_columns.issubset(df.columns):
                st.error(f"❌ Database missing required columns. Found: {list(df.columns)}")
                return
            
            if 'main_df' not in st.session_state or st.session_state.get('last_db_path') != str(parquet_path):
                st.session_state.main_df = df.copy()
                st.session_state.last_db_path = str(parquet_path)
            else:
                df = st.session_state.main_df.copy()
            
            df['exists'] = df['image_path'].apply(lambda x: Path(x).exists())
            images_exist = df['exists'].sum()
            images_missing = len(df) - images_exist
            
            # Check for datetime columns
            has_created_at = 'created_at' in df.columns
            has_modified_at = 'modified_at' in df.columns
            
            st.markdown("---")
            st.markdown("### 📊 Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Entries", len(df))
                st.metric("Images Found", images_exist)
            with col2:
                st.metric("Images Missing", images_missing)
                st.metric("Unique Prompts", df['prompt'].nunique())
            
            # Show datetime statistics if available
            if has_created_at:
                valid_created = df['created_at'].notna().sum()
                if valid_created > 0:
                    oldest = df['created_at'].min()
                    newest = df['created_at'].max()
                    st.caption(f"📅 Date range: {format_datetime(oldest)} to {format_datetime(newest)}")
            
            file_size = parquet_path.stat().st_size / 1024
            if file_size > 1024:
                file_size_str = f"{file_size/1024:.2f} MB"
            else:
                file_size_str = f"{file_size:.2f} KB"
            
            st.caption(f"💾 Database size: {file_size_str}")
            
            st.markdown("---")
            st.info("✏️ Click 'Edit' on any description to modify it. Changes are saved to the database.")
            
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
            
            items_per_page = st.selectbox(
                "Items per page",
                [5, 10, 20, 50, 100],
                index=1,
                key="items_per_page"
            )
            
            st.markdown("---")
            st.markdown("### 🔍 Filters")
            
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
            
            unique_prompts = sorted(df['prompt'].unique())
            if len(unique_prompts) > 1:
                selected_prompts = st.multiselect(
                    "Filter by Prompt",
                    options=["All"] + unique_prompts,
                    default=["All"]
                )
                
                if "All" not in selected_prompts:
                    filtered_df = filtered_df[filtered_df['prompt'].isin(selected_prompts)]
            
            st.markdown("---")
            st.markdown("### 📑 Sorting")
            
            # Build sort options dynamically based on available columns
            sort_options = []

            # Add created date options first if available
            if has_created_at:
                sort_options.extend([
                    "Created Date (Newest First)",
                    "Created Date (Oldest First)"
                ])

            # Then add other options
            sort_options.extend([
                "Image Name (A-Z)",
                "Image Name (Z-A)",
                "Prompt (A-Z)",
                "Prompt (Z-A)"
            ])

            if has_modified_at:
                sort_options.extend([
                    "Modified Date (Newest First)",
                    "Modified Date (Oldest First)"
                ])

            sort_option = st.selectbox(
                "Sort by",
                sort_options,
                index=0  # Will default to "Created Date (Newest First)" if has_created_at is True
            )

            
            filtered_df = apply_sorting(filtered_df, sort_option)
            
            st.markdown("---")
            st.markdown("### 🔎 Search")
            
            search_in = st.selectbox(
                "Search in",
                ["Filename OR Description", "Description", "Filename", "Full Path", "Prompt", "All"],
                index=0,
                help="Choose where to search."
            )
            
            search_query = st.text_input("Search", "", placeholder="Enter search term...")
            
            if search_in == "Filename OR Description":
                st.caption("🔍 Will match if found in filename OR description")
            elif search_in == "All":
                st.caption("🔍 Will search across all fields")
            
            filtered_df = apply_search_filter(filtered_df, search_query, search_in)
            
            if search_query:
                st.info(f"Found {len(filtered_df)} matching result(s)")
            
            # Reset page only when filters actually change
            current_filter = (existence_filter, sort_option, search_query, search_in)
            
            if 'last_filter' not in st.session_state:
                st.session_state.last_filter = current_filter
            elif st.session_state.last_filter != current_filter:
                st.session_state.current_page = 1
                st.session_state.last_filter = current_filter
            else:
                st.session_state.last_filter = current_filter
        else:
            st.info("👈 Enter a database path to get started")
            return
    
    if filtered_df.empty:
        st.info("No entries match the current filter criteria")
        return
    
    # Removed the white box that was here before
    
    total_pages = (len(filtered_df) - 1) // items_per_page + 1
    
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1
    
    current_page = st.session_state.current_page
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_df))
    
    # Simple caption instead of info box
    st.caption(f"Showing items {start_idx + 1}-{end_idx} of {len(filtered_df)}")
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    for idx, row in page_df.iterrows():
        display_image_with_description(row, idx, thumbnail_size)
    
    render_pagination(current_page, total_pages)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Image Gallery Viewer | Built with Streamlit | Powered by Parquet"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
