# Mini Captioner

Mini Captioner is a two-part Python toolset for generating and viewing image descriptions using AI. It consists of a command-line tool to process images and a web-based gallery viewer to display them.

## Features

-   **AI-Powered Captioning**: Uses OpenAI-compatible vision models to generate descriptions for your images.
-   **Batch Processing**: Process an entire directory of images with a single command.
-   **Idempotent**: Avoids re-processing images that already have a description.
-   **Web Gallery Viewer**: A user-friendly Streamlit application to view images alongside their generated descriptions.
-   **Easy to Use**: Simple command-line interface and an intuitive web UI.

## Components

### 1. Captioner CLI (`captioner.py`)

This command-line tool is responsible for processing a directory of images. For each image, it calls an OpenAI-compatible vision model to generate a descriptive caption and saves it as a `.txt` file with the same name as the image.

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
-   `--output`, `-o`: (Optional) The directory to save the output `.txt` files. Defaults to the same directory as the images.
-   `--override`: (Optional) If set, the script will re-process all images, even if they already have a description file.

### 2. Streamlit Viewer (`streamlit_viewer.py`)

This is a web-based application that provides a gallery view of your images and their corresponding descriptions.

#### Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run streamlit_viewer.py
    ```

2.  **Open your browser** to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Enter the path** to your image directory in the sidebar to view the gallery.

**Features:**

-   Side-by-side view of images and their descriptions.
-   Copy and download descriptions.
-   Filter images (e.g., show only images with descriptions).
-   Sort images by name or date.
-   Search within descriptions.
-   Statistics about your image collection.

## Workflow

1.  Place all your images in a single directory.
2.  Use the `captioner.py` script to generate the description files for all images.
3.  Run the `streamlit_viewer.py` app to view your images and their newly created descriptions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
