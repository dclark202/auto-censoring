# FSP Finder

Available for use on [Hugging Face Spaces](https://huggingface.co/spaces/dac202/fsp-finder)! 

Disclaimer: We're working on getting more computational power, using the free CPU provided by Hugging Face it takes about an hour to process a song

## About this tool
**FSP (Foul Speech Pattern) Finder** is an AI-powered explicit content detector and automatic censoring tool useful for preparing music files for radio airplay. We use a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to detect explicit content in music files. Vocals stems are split using [demucs](https://github.com/facebookresearch/demucs) and muted at the identified times to produce an edited file suitable for the air. 

This tool can process files one at a time or in batches. The web interface allows the user to view the full transcript of each track along with the words that will be censored. Additionally, you'll get a link to the [Genius](https://genius.com/) entry for the lyrics of the track, along with a similarity score ([MER](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html)), for cross referencing accuracy. 

## Requirements 
- `pip install -r requirements.txt` to install the necessary dependencies
- [`ffmpeg`](https://ffmpeg.org/) (for handling mp3 files)
- A [Genius API key](https://genius.com/api-clients). This key should be placed in the `GENIUS_API_TOKEN` variable in `fsp.py` (or set as `GENIUS_API_TOKEN` in your system environment).

To start the web interface locally execute `python app.py` in the project directory. On first execution `app.py` will convert the configuration files in `./lora_config` to a full Whisper model stored at `./whisper-medium-ft`.

**Note**: Running this app locally in any reasonable amount of time will require a CUDA enabled GPU with a minimum of 12GB of VRAM (recommended 16GB or more).


## Training and methodology
We trained OpenAI/whisper-medium.en, and english-only automatic speech recognition model, on a portion of the [DALI Dataset](https://zenodo.org/records/2577915). We identified tracks in the DALI dataset that were (1) in English, (2) had a working link to YouTube to grab the audio file. DALI contains timestamped transcriptions of the tracks. We first separated the vocals-only stem from each track, then from that stem extracted only the segments identified in the DALI dataset "lines" entries, i.e., short (5-10s) clips from each track identified as having vocals present. These audio files were saved as mono `.wav` files with a 16 kHz sample rate. 

From this training data we extracted only those lines that a fine-tuned toxicity version of the [cardiffnlp toxicity classifier](https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel) identified as being explicit. This resulted in a dataset of roughly 2000 audio chunks with timestamps. We split this dataset into train/val/test sets, and trained both (1) a LoRA adapter, and (2) the final LM (also called *proj_out*) layer of our Whisper model. Our fine-tuned Whisper model decreases the [match error rate](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html) (MER) of the test set from 0.58424 to 0.48113 (with a similar decrease in [word error rate](https://en.wikipedia.org/wiki/Word_error_rate) from 0.64305 to 0.52992). These error rates don't tell the whole story though: by specifically training on explicit content our model has become very sensitive to explicit content, with the ultimate goal of minimizing false negatives.

Training notebooks for creating the audio files and metdata for DALI, along with preparing the Whisper dataset, and fine-tuning the model can all be found in the `./notebooks` folder. In addition, the notebooks found in `./notebooks/line-dataset-normalizer` played a crucial role in cleaning our data: the lyrics transcriptions in the DALI dataset often contained spelling error, unneccesary spaces and word concatenations, or other punctuation which prevented Whisper from correctly identifying the transcript. 

## Future implementation
- Our main priority is to implement the ability for the user to add their own words to the list of words to be censored by highlighting additional words in the full transcript provided by the model
- We also will continue to improve our models performance by training on larger sets of data. Finding adequate training data is one of the biggest challenges for improving our performance. 

## Credits
- This project was completed as part of the [Erdos Institute's](https://www.erdosinstitute.org/) Deep Learning Bootcamp in Summer 2025.
- All training data comes from the [DALI Dataset](https://zenodo.org/records/2577915).


