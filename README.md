# FSP Finder

**FSP (Foul Speech Pattern) Finder** is an AI-powered explicit content detector and automatic censoring tool useful for preparing music files for radio airplay. We use a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to detect explicit content in music files. Vocals stems are split using [demucs](https://github.com/facebookresearch/demucs) and muted at the identified times to produce an edited file suitable for the air. 

This tool can process files one at a time or in batches. The webapp allows the user to view the full transcript of each track along with the time and words that were automatically censored. Additionally, you'll get a link to the [Genius](https://genius.com/) entry for the lyrics of the track, along with a similarity score ([MER](https://lightning.ai/docs/torchmetrics/stable/text/match_error_rate.html)), for cross referencing accuracy. 


## Requirements 
- `pip install -r requirements.txt` to install the necessary dependencies
- [`ffmpeg`](https://ffmpeg.org/) for handling mp3 files
- A [Genius API key](https://genius.com/api-clients). This key should be placed it in the `GENIUS_API_TOKEN` variable in `fsp.py` (or set as `GENIUS_API_TOKEN` in your system environment)

## Running the app
- To start the web interface locally run `python app.py`. (On first execution `app.py` will convert the configuration files in `lora_config` to a full Whisper model stored at `whisper-medium-ft`) 
- Remote hosting via huggingface (coming soon!)

**Note**: Running this app locally in any reasonable amount of time will require a CUDA enabled GPU with a minimum of 12GB of VRAM (recommended 16GB or more)

## Credits
- Training data comes from the [DALI Dataset](https://zenodo.org/records/2577915)


