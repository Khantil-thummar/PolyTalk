"""
Configuration settings for the voice translation application.
"""

# Model configuration
MODEL_NAME = "facebook/seamless-m4t-v2-large"  # HuggingFace model name
TARGET_LANGUAGE = "hin"  # Target language code (e.g., 'hin' for Hindi)

# Audio processing configuration
COMBINE_CHUNKS = 2  # Number of audio chunks to combine before processing
SAVE_CHUNKS = False  # Whether to save raw audio chunks to disk (in CHUNK_DIR)
SAVE_TRANSLATED_CHUNKS = False  # Whether to save translated audio chunks to disk (in OUTPUT_DIR)
STORE_LIPSYNC_VIDEO = True  # Whether to save generated lipsynced video chunks (in 'results')

# Directory configuration
CHUNK_DIR = "chunks"  # Directory to store original audio chunks
OUTPUT_DIR = "converted_chunks"  # Directory to store translated audio files

# Server configuration
HTML_TEMPLATE_DIR = "static"  # Directory for HTML templates/static files