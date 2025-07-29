## Requirements 
- `pip install -r requirements.txt` to install the necessary packages
- Download the state dictionary files from [here](https://drive.google.com/drive/folders/1ptaOvIyqDhAf8wgF__Td6PwpP5vr89vJ?usp=sharing) and save to the `/data/` folder. Then, execute `python create-models.py` to create the fine tuned models
- Last you'll need a [Genius API key](https://genius.com/api-clients). Place it in the `GENIUS_API_TOKEN` variable in `fsp.py`

## Running the app

`python app.py` to start the web interface

# Warning

Running this app locally in any reasonable amount of time will require a CUDA capable GPU with at least 16GB of VRAM
