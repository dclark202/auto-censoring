# FSP Finder

**FSP (Foul Speech Pattern) Finder** is an AI-powered explicit content detector and automatic censoring tool useful for preparing music files for radio airplay. We use a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to detect explicit content in music files. The vocals are then muted at the indentified times, by using [demucs](https://github.com/facebookresearch/demucs) to first extract the vocals-only stem. The muted vocals track is then combined with the instruments stem to produce an edited music track ready for radio airplay. 

This tool can process files one at a time or in batches. The webapp allows the user to view the full transcript of each track along with the time and words that were automatically censored. Additionally, you'll get a link to the [Genius](https://genius.com/) entry for the lyrics of the track, along with a similarity score, for cross referencing accuracy. 


## Requirements 
- `pip install -r requirements.txt` to install the necessary packages
- Run `python create-models.py` in the base repo folder to create the fine-tuned models
- Obtain a [Genius API key](https://genius.com/api-clients) and place it in the `GENIUS_API_TOKEN` variable in `fsp.py`

## Running the app
- To start the web interface locally run `python app.py`. (Warning: running this app locally in any reasonable amount of time requires a CUDA enabled GPU with a minimum of 12GB of VRAM (recommended 16GB+)
- Remote hosting via huggingface (coming soon!)

## Credits
- Training data comes from the [DALI Dataset](https://zenodo.org/records/2577915)


