import torch
from transformers import WhisperForConditionalGeneration

# 1. Load the original pre-trained base model from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

# 2. Load your saved final layer weights
final_layer_weights = torch.load('proj_out_ft.pt')

# 3. Apply your fine-tuned weights to the base model's final layer
model.proj_out.load_state_dict(final_layer_weights)

# 4. Save the model
model.save_pretrained('../whisper-medium-ft')