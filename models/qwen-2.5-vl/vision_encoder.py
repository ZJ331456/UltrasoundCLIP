import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch.nn.functional as F

def extract_vision_features(model_path, image_path):
    """
    Extract vision features from an image using Qwen2.5-VL vision encoder
    
    Args:
        model_path: Path to the Qwen2.5-VL model
        image_path: Path to the input image
        
    Returns:
        image_embeddings: torch.Tensor of shape [num_patches, hidden_size]
        image_grid_thw: torch.Tensor containing temporal, height, width info
    """
    
    # Load the full model first (we'll extract the vision part)
    print("Loading Qwen2.5-VL model...")
    full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"  # Auto-detects GPU if available
    )

    # Extract the vision encoder from the full model
    vision_model = full_model.model.visual  # This is the vision transformer
    print(f"Vision model loaded on device: {next(vision_model.parameters()).device}")

    # Load the processor for image handling
    processor = AutoProcessor.from_pretrained(model_path)

    # Load and process the image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    
    # Process the image into pixel_values (ready for the vision encoder)
    # For Qwen2.5-VL, we need to provide text input as well, even if we only want vision features
    dummy_text = "<|vision_start|><|image_pad|><|vision_end|>"
    inputs = processor(text=[dummy_text], images=[image], return_tensors="pt")

    # Move inputs to the same device as the model
    device = next(vision_model.parameters()).device
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs["image_grid_thw"].to(device)

    # Run the vision encoder to get embeddings
    print("Extracting vision features...")
    with torch.no_grad():  # For inference without gradients
        # Use the vision model's forward method directly
        image_embeddings = vision_model(pixel_values, grid_thw=image_grid_thw)

    return image_embeddings, image_grid_thw

if __name__ == "__main__":
    model_name = "/media/ps/data-ssd/DolphinV1/dolphinV1_3B"
    image_path = "/media/ps/data-ssd/UltrasoundRAG/CLIP/models/qwen-2.5-vl/甲状腺超声图.jpg"
    
    # Extract vision features
    image_embeddings, image_grid_thw = extract_vision_features(model_name, image_path)
    
    # Print results
    print(f"\nResults:")
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Image embedding shape: {image_embeddings}")
    print(f"Image grid (t,h,w): {image_grid_thw}")
    print(f"Feature dimension: {image_embeddings.shape[-1]}")
    print(f"Number of patches: {image_embeddings.shape[0]}")
    
    # Save the features if needed
    # torch.save(image_embeddings, "image_features.pt")
    
    print("\nVision features extracted successfully!")
    print("You can now use 'image_embeddings' as your image encoder output for downstream tasks.")