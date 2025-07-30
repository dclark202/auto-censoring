import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration
from transformers import AutoModelForSequenceClassification

print('Creating Whisper model...')
# 1. Load the base model
base_model_name = "openai/whisper-medium.en"
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)

# 2. Load the LoRA adapter onto the base model
lora_config_dir = "./lora-config"
model = PeftModel.from_pretrained(base_model, lora_config_dir)

# 3. Merge to the whisper model and save the full thing
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./whisper-medium-ft")
print('Done.')

########## 

print('Creating toxicity classifier...')

model_name = "cardiffnlp/twitter-roberta-large-sensitive-multilabel"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model_state_dict = torch.load('./data/state_dict_tox.pt')
model.load_state_dict(model_state_dict)

model.save_pretrained('./roberta_search')
print('Done.')

print('Ready to go!')