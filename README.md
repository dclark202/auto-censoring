## Available for use at our website [www.fspfinder.com](www.fspfinder.com)

Watch the project video by clicking on the image below (Note: some of the material in this video is now outdated with the release of the webite.)

[![Watch the project video](https://img.youtube.com/vi/csp4E_csyco/0.jpg)](https://www.youtube.com/watch?v=csp4E_csyco)



## About this tool
**FSP Finder** is an AI-powered foul speech pattern (FSP) detector and automatic censoring tool useful for preparing music files for radio airplay. We use a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to transcribe the lyrics of uploaded music files (with word timestamps). First, [demucs](https://github.com/facebookresearch/demucs) is used to split the vocals stem from the track. Profanity, racial slurs, and other "always explicit" content is immediately flagged to be muted. The user is then presented with the transcript and can select additional words to be muted.

This tool can process files one at a time or in batches. The interface allows the user to view the full transcript of each track along with the words that will be censored. Additionally, you'll get a link to the [Genius](https://genius.com/) entry for the lyrics of the track, along with a similarity score ([MER](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html)), for cross referencing accuracy. 

## Requirements 
- `pip install -r requirements.txt` to install the necessary dependencies (recommend installing PyTorch separately with GPU support)
- [`ffmpeg`](https://ffmpeg.org/) (for handling mp3 files)
- (Optional) [Genius API key](https://genius.com/api-clients), set as `GENIUS_API_TOKEN` in your system environment.

## Running FSP Finder locally
- To start the backend: `uvicorn backend:app`
- To start the frontend `flask --app app-flask run`

On first execution the configuration files in `./lora_config` will be converted to a full Whisper model stored at `./whisper-medium-ft` (the full Whisper model is necessary for using [Whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) to produce word timestamps). Please note, running this app locally in any reasonable amount of time will require a CUDA enabled GPU with a minimum of 12GB of VRAM.

## Training and methodology
We trained OpenAI/whisper-medium.en, and english-only automatic speech recognition model, on a portion of the [DALI Dataset](https://zenodo.org/records/2577915). We identified tracks in the DALI dataset that were (1) in English, (2) had a working link to YouTube to grab the audio file. DALI contains timestamped transcriptions of the tracks. We first separated the vocals-only stem from each track, then from that stem extracted only the segments identified in the DALI dataset "lines" entries, i.e., short (5-10s) clips from each track identified as having vocals present. These audio files were saved as mono `.wav` files with a 16 kHz sample rate. 

From this training data we extracted only those lines that a fine-tuned toxicity version of the [cardiffnlp toxicity classifier](https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel) identified as being explicit. This resulted in a dataset of roughly 2000 audio chunks with timestamps. We split this dataset into train/val/test sets, and trained both (1) a LoRA adapter, and (2) the final LM (also called *proj_out*) layer of our Whisper model. Our fine-tuned Whisper model decreases the [match error rate](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html) (MER) of the test set from 0.58424 to 0.48113 (with a similar decrease in [word error rate](https://en.wikipedia.org/wiki/Word_error_rate) from 0.64305 to 0.52992). These error rates don't tell the whole story though: by specifically training on explicit content our model has become very sensitive to explicit content, with the ultimate goal of minimizing false negatives (i.e., maximizing recall).

Training notebooks for creating the audio files and metdata for DALI, along with preparing the Whisper dataset, and fine-tuning the model can all be found in the `./dev/notebooks` folder. In addition, the notebooks found in `./dev/notebooks/line-dataset-normalizer` played a crucial role in cleaning our data: the lyrics transcriptions in the DALI dataset often contained spelling error, unneccesary spaces and word concatenations, or other punctuation which prevented Whisper from correctly identifying the transcript. The master lists containg the relevant metadata (filename, transcript, etc.) for each of the train/val/test sets is contained in `./dev/data`.

## Future implementation
We have a pipeline for using a language model ([Google Gemma 2](https://huggingface.co/google/gemma-2-9b))  to detect edge case explicit content (drug references, violence, etc.). This is temporarily disabled on the website, as we collect data for fine-tuning the LLM first. 

## Credits
- This project was completed as part of the [Erdos Institute](https://www.erdosinstitute.org/)'s Deep Learning Bootcamp in Summer 2025.
- All training data comes from the [DALI Dataset](https://zenodo.org/records/2577915).


