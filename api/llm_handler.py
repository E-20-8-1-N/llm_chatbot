import os
from langchain_community.llms import LlamaCpp
from PIL import Image
import exifread
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
# These will be set by environment variables in your TrueNAS YAML
LLM_MODEL_PATH_IN_CONTAINER = os.getenv("LLM_MODEL_PATH", "/models/llama-2-7b.Q4_0.gguf") # Default, override with ENV
USER_DATA_ROOT_IN_CONTAINER = os.getenv("USER_DATA_PATH_IN_CONTAINER", "/data") # Default, override with ENV
N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1")) # -1 for all, 0 for CPU, or specific number
N_CTX = int(os.getenv("LLAMA_N_CTX", "4096"))
# --- End Configuration ---

llm = None

def initialize_llm():
    global llm
    if not os.path.exists(LLM_MODEL_PATH_IN_CONTAINER):
        logger.error(f"LLM Model not found at {LLM_MODEL_PATH_IN_CONTAINER}. LLM will not function.")
        llm = None
        return

    try:
        logger.info(f"Initializing LlamaCpp model from: {LLM_MODEL_PATH_IN_CONTAINER}")
        logger.info(f"Using n_gpu_layers: {N_GPU_LAYERS}, n_ctx: {N_CTX}")
        llm = LlamaCpp(
            model_path=LLM_MODEL_PATH_IN_CONTAINER,
            n_gpu_layers=N_GPU_LAYERS,  # Offload all/some layers to GPU. For iGPU, might need to be conservative.
                                        # Start with a small number (e.g., 10-20) or 0 if full offload fails.
            n_batch=512,                # Should be fine.
            n_ctx=N_CTX,                # Context window.
            f16_kv=True,                # Use FP16 for K/V cache.
            verbose=True,               # For Llama.cpp logging.
            # temperature=0.7,
        )
        logger.info("LLM initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing LlamaCpp: {e}", exc_info=True)
        llm = None

# Call initialization once when the module is loaded
initialize_llm()

def get_file_metadata(file_path_in_container):
    """
    Extracts metadata from a given file.
    Adjust this function based on the types of files and metadata you need.
    """
    metadata_parts = []
    try:
        if not os.path.exists(file_path_in_container):
            return f"Error: File not found at '{file_path_in_container}'."

        stat_info = os.stat(file_path_in_container)
        metadata_parts.append(f"File Name: {os.path.basename(file_path_in_container)}")
        metadata_parts.append(f"File Size: {stat_info.st_size} bytes")
        metadata_parts.append(f"Last Modified: {datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Attempt to get created date (might be platform dependent or same as mtime)
        try:
            metadata_parts.append(f"Created Date: {datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
        except AttributeError: # Windows might have st_birthtime
             metadata_parts.append(f"Created Date (birthtime): {datetime.fromtimestamp(stat_info.st_birthtime).strftime('%Y-%m-%d %H:%M:%S')}")


        file_ext = os.path.splitext(file_path_in_container)[1].lower()

        if file_ext in ['.jpg', '.jpeg', '.tiff', '.heic', '.png']:
            try:
                with open(file_path_in_container, 'rb') as f_img:
                    tags = exifread.process_file(f_img, details=False)
                    if tags:
                        metadata_parts.append("EXIF Data:")
                        for tag, value in tags.items():
                            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                                # Shorten long byte arrays for display
                                val_str = str(value)
                                if len(val_str) > 100:
                                    val_str = val_str[:100] + "..."
                                metadata_parts.append(f"  - {tag}: {val_str}")
                    else:
                         metadata_parts.append("No EXIF data found.")
            except Exception as e_exif:
                metadata_parts.append(f"Could not read EXIF data: {e_exif}")

            try:
                with Image.open(file_path_in_container) as img:
                    metadata_parts.append(f"Image Dimensions: {img.width}x{img.height}")
                    metadata_parts.append(f"Image Format: {img.format}")
            except Exception as e_pil:
                 metadata_parts.append(f"Could not read image properties with Pillow: {e_pil}")


    except Exception as e:
        logger.error(f"Error getting metadata for {file_path_in_container}: {e}", exc_info=True)
        return f"Error processing file {os.path.basename(file_path_in_container)}: {e}"

    return "\n".join(metadata_parts)


def process_query_with_llm(user_question: str, relative_file_path: str):
    if llm is None:
        logger.error("LLM not initialized. Cannot process query.")
        return "Error: The Language Model is not available. Please check the server logs."

    full_file_path_in_container = os.path.join(USER_DATA_ROOT_IN_CONTAINER, relative_file_path.lstrip('/'))
    logger.info(f"Processing query for file: {full_file_path_in_container}")

    file_info = get_file_metadata(full_file_path_in_container)
    if "Error:" in file_info: # Propagate file processing errors
        return file_info

    prompt_template = f"""
You are a helpful AI assistant. You will be given some information about a file and a user's question.
Answer the user's question based *only* on the provided file information.
If the information is not available in the provided details, state that clearly.

File Information:
---
{file_info}
---

User's Question: {user_question}

Answer:
"""
    logger.debug(f"Generated prompt for LLM:\n{prompt_template}")

    try:
        response = llm.invoke(prompt_template)
        logger.info(f"LLM response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error during LLM inference: {e}", exc_info=True)
        return f"Error processing your question with the LLM: {e}"