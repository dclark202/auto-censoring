# Available for use at our [website](https://www.fspfinder.com)!

Watch the project video by clicking on the image below (Note: some of the material in this video is now outdated with the release of the webite.)

[![Watch the project video](https://img.youtube.com/vi/csp4E_csyco/0.jpg)](https://www.youtube.com/watch?v=csp4E_csyco)



## About this tool
**FSP Finder** is an AI-powered foul speech pattern (FSP) detector and automatic censoring tool useful for preparing music files for radio airplay. We use a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to transcribe the lyrics of uploaded music files (with word timestamps). First, [demucs](https://github.com/facebookresearch/demucs) is used to split the vocals stem from the track. Profanity, racial slurs, and other "always explicit" content is immediately flagged to be muted. The user is then presented with the transcript and can select additional words to be muted.

This tool can process files one at a time or in batches. The interface allows the user to view the full transcript of each track along with the words that will be censored. Additionally, you'll get a link to the [Genius](https://genius.com/) entry for the lyrics of the track, along with a similarity score ([MER](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html)), for cross referencing accuracy. 

## Prerequisites

Before you begin, ensure you have the following software installed on your system:
* **Git:** For cloning the repository.
* **Docker Desktop:** The entire application runs inside Docker containers. [Download Docker Desktop here](https://www.docker.com/products/docker-desktop/).

### Local Installation & Setup

Follow these steps to get the application running on your local machine.

#### 1. Clone the Repository
Open your terminal and clone the project files from GitHub.
```bash
git clone https://github.com/dclark202/auto-censoring.git
cd auto-censoring
```

#### 2. Configure Your Secret API Keys
The application requires API keys for [Hugging Face](https://huggingface.co/) and [Genius](http://genius.com). You'll need to create a `.env` file to store them.

1.  Copy the example file to a new `.env` file.
    ```bash
    # On Windows (Command Prompt)
    copy .env.example .env

    # On macOS/Linux
    cp .env.example .env
    ```
2.  Open the new `.env` file in a text editor and replace the placeholder values with your actual secret keys.

### Running the Application

This project includes a convenient launcher script that builds the Docker containers, starts the services, and opens the application in your browser automatically.

From your terminal in the project's root directory, run:
```bash
python start.py
```
The first time you run this, Docker will need to download several gigabytes of data for the base images and Python packages, which may take a significant amount of time. Subsequent launches will be much faster.

Once the startup is complete, your default web browser will open to `http://12.0.0.1:7860`, and you can begin using the application.

### Stopping the Application

To stop all the running Docker containers, open a new terminal in the project directory and run:
```bash
docker compose down
```

### Note 

On first execution the configuration files in `./lora_config` will be converted to a full Whisper model stored at `./whisper-medium-ft` (the full Whisper model is necessary for using [Whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) to produce word timestamps). Please note, running this app locally in any reasonable amount of time will require a CUDA enabled GPU with a minimum of 12GB of VRAM.

# Training and methodology
We trained OpenAI/whisper-medium.en, and english-only automatic speech recognition model, on a portion of the [DALI Dataset](https://zenodo.org/records/2577915). We identified tracks in the DALI dataset that were (1) in English, (2) had a working link to YouTube to grab the audio file. DALI contains timestamped transcriptions of the tracks. We first separated the vocals-only stem from each track, then from that stem extracted only the segments identified in the DALI dataset "lines" entries, i.e., short (5-10s) clips from each track identified as having vocals present. These audio files were saved as mono `.wav` files with a 16 kHz sample rate. 

From this training data we extracted only those lines that a fine-tuned toxicity version of the [cardiffnlp toxicity classifier](https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel) identified as being explicit. This resulted in a dataset of roughly 2000 audio chunks with timestamps. We split this dataset into train/val/test sets, and trained both (1) a LoRA adapter, and (2) the final LM (also called *proj_out*) layer of our Whisper model. Our fine-tuned Whisper model decreases the [match error rate](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html) (MER) of the test set from 0.58424 to 0.48113 (with a similar decrease in [word error rate](https://en.wikipedia.org/wiki/Word_error_rate) from 0.64305 to 0.52992). These error rates don't tell the whole story though: by specifically training on explicit content our model has become very sensitive to explicit content, with the ultimate goal of minimizing false negatives (i.e., maximizing recall).

Training notebooks for creating the audio files and metdata for DALI, along with preparing the Whisper dataset, and fine-tuning the model can all be found in the `./dev/notebooks` folder. In addition, the notebooks found in `./dev/notebooks/line-dataset-normalizer` played a crucial role in cleaning our data: the lyrics transcriptions in the DALI dataset often contained spelling error, unneccesary spaces and word concatenations, or other punctuation which prevented Whisper from correctly identifying the transcript. The master lists containg the relevant metadata (filename, transcript, etc.) for each of the train/val/test sets is contained in `./dev/data`.

## Credits
- This project was completed as part of the [Erdos Institute](https://www.erdosinstitute.org/)'s Deep Learning Bootcamp in Summer 2025.
- All training data comes from the [DALI Dataset](https://zenodo.org/records/2577915).


