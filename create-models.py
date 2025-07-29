import torch
from transformers import WhisperForConditionalGeneration
from transformers import AutoModelForSequenceClassification

print('Creating Whisper model...')
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")

final_layer_weights = torch.load('./data/proj_out_ft.pt')
model.proj_out.load_state_dict(final_layer_weights)

model.save_pretrained('./whisper-medium-ft')
print('Done.')

print('Creating toxicity classifier...')

model_name = "cardiffnlp/twitter-roberta-large-sensitive-multilabel"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model_state_dict = torch.load('./data/state_dict_tox.pt')
model.load_state_dict(model_state_dict)

model.save_pretrained('./roberta_search')
print('Done.')

print('Ready to go!')