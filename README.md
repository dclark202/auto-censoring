## About

This repository contains training and methodology files for FSP Finder. The final product, including code for hosting a local version of the web app, can be found [here](https://github.com/dclark202/fsp-finder). The full application can be used at our website [www.fspfinder.com](https://www.fspfinder.com).

Watch the project video by clicking on the image below (Note: some of the material in this video is now outdated with the release of the webite.)

[![Watch the project video](https://img.youtube.com/vi/csp4E_csyco/0.jpg)](https://www.youtube.com/watch?v=csp4E_csyco)


## Training and methodology
We trained OpenAI/whisper-medium.en, and english-only automatic speech recognition model, on a portion of the [DALI Dataset](https://zenodo.org/records/2577915). We identified tracks in the DALI dataset that were (1) in English, (2) had a working link to YouTube to grab the audio file. DALI contains timestamped transcriptions of the tracks. We first separated the vocals-only stem from each track, then from that stem extracted only the segments identified in the DALI dataset "lines" entries, i.e., short (5-10s) clips from each track identified as having vocals present. These audio files were saved as mono `.wav` files with a 16 kHz sample rate. 

From this training data we extracted only those lines that a fine-tuned toxicity version of the [cardiffnlp toxicity classifier](https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel) identified as being explicit. This resulted in a dataset of roughly 2000 audio chunks with timestamps. We split this dataset into train/val/test sets, and trained both (1) a LoRA adapter, and (2) the final LM (also called *proj_out*) layer of our Whisper model. Our fine-tuned Whisper model decreases the [match error rate](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html) (MER) of the test set from 0.58424 to 0.48113 (with a similar decrease in [word error rate](https://en.wikipedia.org/wiki/Word_error_rate) from 0.64305 to 0.52992). These error rates don't tell the whole story though: by specifically training on explicit content our model has become very sensitive to explicit content, with the ultimate goal of minimizing false negatives (i.e., maximizing recall).

Training notebooks for creating the audio files and metdata for DALI, along with preparing the Whisper dataset, and fine-tuning the model can all be found in the `./dev/notebooks` folder. In addition, the notebooks found in `./dev/notebooks/line-dataset-normalizer` played a crucial role in cleaning our data: the lyrics transcriptions in the DALI dataset often contained spelling error, unneccesary spaces and word concatenations, or other punctuation which prevented Whisper from correctly identifying the transcript. The master lists containg the relevant metadata (filename, transcript, etc.) for each of the train/val/test sets is contained in `./dev/data`.

## Credits
- This project was completed as part of the [Erdos Institute](https://www.erdosinstitute.org/)'s Deep Learning Bootcamp in Summer 2025.
- All training data comes from the [DALI Dataset](https://zenodo.org/records/2577915).


