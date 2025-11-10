# Mini Captioner

Mini Captioner is a two-part Python toolset for generating and viewing image descriptions using AI. It consists of a command-line tool to process images and save the results to a Parquet database, and a web-based gallery viewer to display the images and their descriptions from the database.

## Features

-   **AI-Powered Captioning**: Uses OpenAI-compatible vision models to generate descriptions for your images.
-   **Batch Processing**: Process an entire directory of images with a single command.
-   **Parquet Database**: Stores all image paths, prompts, and descriptions in a single, efficient Parquet file.
-   **Idempotent**: Avoids re-processing images that already have an entry in the database (unless overridden).
-   **Graceful Exit**: Saves progress automatically if interrupted (Ctrl-C).
-   **Web Gallery Viewer**: A user-friendly Streamlit application to browse, search, and filter images and their generated descriptions.
-   **Easy to Use**: Simple command-line interface and an intuitive web UI.

## Components

### 1. Captioner CLI (`captioner.py`)

This command-line tool processes a directory of images, generates a description for each using a vision model, and saves the results to a Parquet database file.

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

```bash
python captioner.py -d /path/to/your/images -p "A detailed description of this image is:"
```

**Arguments:**

-   `--directory`, `-d`: (Required) The directory containing the images to process.
-   `--prompt`, `-p`: (Required) The prompt to send to the vision model for each image.
-   `--database`, `--db`: (Optional) The path to the Parquet database file. Defaults to `vision_ai.parquet` in the current directory.
-   `--override`: (Optional) If set, the script will re-process all images and update their existing entries in the database. By default, it skips images that are already in the database.

### 2. Streamlit Viewer (`streamlit_viewer.py`)

This is a web-based application that provides a gallery view of your images and their corresponding descriptions by reading from the Parquet database.

#### Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run streamlit_viewer.py
    ```

2.  **Open your browser** to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Enter the path** to your Parquet database file in the sidebar to view the gallery. The default path is `./vision_ai.parquet`.

**Features:**

-   Side-by-side view of images and their descriptions.
-   Copy and download descriptions.
-   Filter images by existence, prompt, or search query.
-   Sort images by name or prompt.
-   Search within descriptions, image paths, or prompts.
-   Statistics about your image collection and database.
-   Pagination for large collections.

## Workflow

1.  Place all your images in a single directory.
2.  Use the `captioner.py` script to generate descriptions. This will create a `vision_ai.parquet` file (or a custom-named one if you use the `--database` option).
    ```bash
    python captioner.py -d /path/to/images -p "A photo of"
    ```
3.  Run the `streamlit_viewer.py` app.
    ```bash
    streamlit run streamlit_viewer.py
    ```
4.  In the web interface, make sure the path to the Parquet database is correct. The app will then display your images and their newly created descriptions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.