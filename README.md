# Mini Captioner

Mini Captioner is a two-part Python toolset for generating and viewing image descriptions using AI. It consists of a command-line tool to process images and save the results to a Parquet database, and a web-based gallery viewer to display the images and their descriptions from the database.

## Features

-   **AI-Powered Captioning**: Uses OpenAI-compatible vision models to generate descriptions for your images.
-   **Batch Processing**: Process an entire directory of images with a single command.
-   **Parquet Database**: Stores all image paths, prompts, and descriptions in a single, efficient Parquet file.
-   **Idempotent**: Avoids re-processing images that already have an entry in the database (unless overridden).
-   **Graceful Exit**: Saves progress automatically if interrupted (Ctrl-C).
-   **Web Gallery Viewer**: A user-friendly Streamlit application to browse, search, and filter images and their generated descriptions.
-   **Edit Descriptions**: Directly edit and save changes to image descriptions within the web viewer.
-   **Easy to Use**: Simple command-line interface and an intuitive web UI.

## Components

### 1. Captioner CLI (`captioner.py`)

This command-line tool processes images from various sources, generates a description for each using a vision model, and saves the results to a Parquet database file. It now includes automatic image resizing to 1 megapixel to optimize performance and cost, and tracks creation and modification timestamps for each entry.

#### Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Create a `.env` file** in the root of the project and add your API credentials:

    ```
    OPENAI_API_BASE="your_api_base_url"
    OPENAI_API_KEY="your_api_key"
    OPENAI_MODEL_NAME="your_model_name"
    ```

#### Usage

The tool offers flexible ways to specify input images:

**Process all images in a directory:**
```bash
python captioner.py -i /path/to/your/images -p "A detailed description of this image is:"
```

**Process a single image file:**
```bash
python captioner.py -i /path/to/image.jpg -p "A photo of"
```

**Process images using a glob pattern (wildcards):**
```bash
python captioner.py -i "images/*.jpg" -p "A close-up shot of"
```

**Process a list of image paths from a text file:**
```bash
python captioner.py -f /path/to/filelist.txt -p "Describe the contents of this image"
```

**Arguments:**

-   `--input`, `-i`: (Required) The input to process. Can be a directory, a single file, or a glob pattern (e.g., `"*.jpg"`).
-   `--file-list`, `-f`: (Required) A text file containing a list of image paths to process, with one path per line.
-   `--prompt`, `-p`: (Required) The prompt to send to the vision model for each image.
-   `--database`, `--db`: (Optional) The path to the Parquet database file. Defaults to `vision_ai.parquet` in the current directory.
-   `--override`: (Optional) If set, the script will re-process all images and update their existing entries in the database. By default, it skips images that are already in the database.
-   `--directory`, `-d`: [DEPRECATED] Use `--input` instead for processing directories.

### 2. Streamlit Viewer (`streamlit_viewer.py`)

This is a web-based application that provides a gallery view of your images and their corresponding descriptions by reading from the Parquet database. It also allows for editing descriptions directly in the UI.

#### Usage

1.  **Run the Streamlit app:**

    To run with the default database path (`./vision_ai.parquet`):
    ```bash
    streamlit run streamlit_viewer.py
    ```

    To specify a database path via the command line:
    ```bash
    streamlit run streamlit_viewer.py -- --database /path/to/your/database.parquet
    ```
    *(Note the `--` which is required to pass arguments to the script itself.)*

2.  **Open your browser** to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  If you didn't specify a database via the CLI, you can **enter the path** to your Parquet database file in the sidebar to view the gallery.

**Features:**

-   **Edit Mode**: Click the "Edit" button on any entry to modify its description. Changes are saved directly back to the Parquet file.
-   Side-by-side view of images and their descriptions.
-   Copy and download descriptions.
-   Filter images by existence, prompt, or search query.
-   Sort images by name or prompt.
-   Search within descriptions, image paths, or prompts.
-   Statistics about your image collection and database.
-   Pagination for large collections.

### 3. Image Directory Finder (`find_image_dirs.py`)

This is a simple utility script that scans a directory recursively and prints a list of all subdirectories that contain at least one image file (`.jpg`, `.jpeg`, `.png`, or `.webp`). This is useful for identifying which folders you might want to process with the main captioner tool.

#### Usage

```bash
python find_image_dirs.py /path/to/your/main_folder
```

**Arguments:**

-   `PATH`: (Required) The root directory path to start scanning from. The script will report all subdirectories under this path that contain images.

## Workflow

1.  Place all your images in a single directory.
2.  Use the `captioner.py` script to generate descriptions. This will create a `vision_ai.parquet` file (or a custom-named one if you use the `--database` option).
    ```bash
    python captioner.py -i /path/to/images -p "A photo of"
    ```
3.  Run the `streamlit_viewer.py` app, optionally pointing it to your database.
    ```bash
    streamlit run streamlit_viewer.py -- --db vision_ai.parquet
    ```
4.  Use the web interface to view, search, and edit your image descriptions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
