"""
Configuration settings for the voice translation application.
"""

# Model configuration
MODEL_NAME = "facebook/seamless-m4t-v2-large"
TARGET_LANGUAGE = "hin"  # Hindi

# Audio processing configuration
COMBINE_CHUNKS = 2  # Number of chunks to combine before processing
SAVE_CHUNKS = True  # Whether to save audio chunks to disk

# Directory configuration
CHUNK_DIR = "chunks"  # Directory to store original audio chunks
OUTPUT_DIR = "converted_chunks"  # Directory to store translated audio files

# Server configuration
HTML_TEMPLATE_DIR = "static" 