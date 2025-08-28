# backend.py
import os
import re
import html
import json
import shutil
import tempfile
import uuid
from datetime import datetime

# FastAPI and related imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ML/Audio libraries
import torch
import transformers
from transformers import AutoTokenizer, WhisperForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import whisper
import whisper_timestamped as whisper_t
import demucs.separate
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
import lyricsgenius
import jiwer

# Hide excessive warning messages from transformers
transformers.logging.set_verbosity_error()

print(f"Executing backend.py at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

###############################################################################################
### CONFIGURATION & CONSTANTS
###############################################################################################

# --- API Keys & Tokens (Load from environment variables for security) ---
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN")
if not GENIUS_API_TOKEN:
    print("WARNING: GENIUS_API_TOKEN environment variable not set. Genius API features will fail.")
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True) if GENIUS_API_TOKEN else None

# --- Model & Device Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LLM_MODEL_ID = "google/gemma-2-9b-it"
WHISPER_BASE_MODEL = "openai/whisper-medium.en"
WHISPER_FT_MODEL_PATH = 'whisper-medium-ft'
LORA_CONFIG_PATH = './lora_config'

# --- Censoring Word Lists ---
# Note: A bunch of curse words and racial slurs are defined below.
default_curse_words = {
    'fuck', 'shit', 'piss', 'bitch', 'nigg', 'dyke', 'cock', 'faggot', 
    'cunt', 'tits', 'pussy', 'dick', 'asshole', 'whore', 'goddam',
    'douche', 'chink', 'tranny', 'jizz', 'kike', 'gook', 'cocksucker'
}
singular_curse_words = {
    'fag', 'cum', 'clit', 'wank' 'ho', 'hoes'
}

###############################################################################################
### CORE LOGIC & HELPER FUNCTIONS
###############################################################################################

def load_whisper_model(model_path, lora_config, base_model_name):
    """Creates the full fine-tuned Whisper model from LoRA weights if it doesn't exist."""
    if os.path.exists(f'./{model_path}/model.safetensors'):
        print(f'Fine-tuned model at {model_path} already exists.')
        return
    
    print(f'Fine-tuned model not found. Creating model from LoRA configuration at {lora_config}')
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, lora_config)
    model = model.merge_and_unload()
    model.save_pretrained(model_path, save_serialization=False)
    print(f'Whisper model from {lora_config} saved at {model_path}')

# --- Start of functions moved from original script ---

# Removes all punctuation and returns lower case only words
def remove_punctuation(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower()

# For silencing the audio tracks at the indicated times
def silence_audio_segment(input_audio_path, output_audio_path, times):
    audio = AudioSegment.from_file(input_audio_path)
    for (start_ms, end_ms) in times:
        silence = AudioSegment.silent(duration=end_ms - start_ms)
        audio = audio[:start_ms] + silence + audio[end_ms:]
    audio.export(output_audio_path, format='wav')

# For combining the vocals and instrument stems once the censoring has been applied
def combine_audio(path1, path2, outpath):
    audio1 = AudioSegment.from_file(path1, format='wav')
    audio2 = AudioSegment.from_file(path2, format='wav')
    combined_audio = audio1.overlay(audio2)
    combined_audio.export(outpath, format="mp3")

# Extracts metadata from the original song
def get_metadata(original_audio_path):
    try:
        audio_orig = EasyID3(original_audio_path)
        metadata = {'title': audio_orig.get('title', [None])[0], 'artist': audio_orig.get('artist', [None])[0], 'album': audio_orig.get('album', [None])[0], 'year': audio_orig.get('date', [None])[0]}
    except Exception:
        metadata = {'title': 'N/A', 'artist': 'N/A', 'album': 'N/A', 'year': 'N/A'}
    return metadata

# Transfers metadata between two songs
def transfer_metadata(original_audio_path, edited_audio_path):
    try:
        audio_orig = EasyID3(original_audio_path)
        audio_edit = EasyID3(edited_audio_path)
        for key in audio_orig.keys():
            audio_edit[key] = audio_orig[key]
        audio_edit.save()
    except Exception as e:
        print(f"Could not transfer metadata: {e}")

# Lookup url on genius of lyrics for given song
def get_genius_url(artist, song_title):
    if not genius or not artist or not song_title or artist == 'N/A' or song_title == 'N/A': return None
    try:
        song = genius.search_song(song_title, artist)
        return song.url if song else None
    except Exception: return None

# It's called calculate_wer but I'm actually using *mer*
def calculate_wer(ground_truth, hypothesis):
    if not ground_truth or not hypothesis or "not available" in ground_truth.lower(): return None
    try:
        transformation = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ExpandCommonEnglishContractions(), jiwer.RemoveEmptyStrings()])
        error = jiwer.mer(transformation(ground_truth), transformation(hypothesis))
        return f"{error:.3f}"
    except Exception: return "Error"

# Gets the lyrics from genius for a given song
def get_genius_lyrics(artist, song_title):
    if not genius or not artist or not song_title or artist == 'N/A' or song_title == 'N/A': return "Lyrics not available (missing metadata or Genius API key)."
    try:
        song = genius.search_song(song_title, artist)
        return song.lyrics if song else "Could not find lyrics on Genius."
    except Exception: return "An error occurred while searching for lyrics."

# Separate track via demucs, evaluate vocals with Whisper
def analyze_audio(audio_path, model, device):
    run_temp_dir = tempfile.mkdtemp()
    source_path = os.path.abspath(audio_path)
    temp_audio_path = os.path.join(run_temp_dir, 'temp_audio.mp3')
    shutil.copy(source_path, temp_audio_path)

    metadata = get_metadata(temp_audio_path)
    metadata['genius_url'] = get_genius_url(metadata['artist'], metadata['title'])
    metadata['genius_lyrics'] = get_genius_lyrics(metadata['artist'], metadata['title'])

    demucs.separate.main(["--two-stems", "vocals", "-n", "mdx_extra", "-o", run_temp_dir, temp_audio_path])
    demucs_out_name = os.path.splitext(os.path.basename(temp_audio_path))[0]
    vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "vocals.wav")
    no_vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "no_vocals.wav")

    audio = whisper_t.load_audio(vocals_path)
    result = whisper_t.transcribe(model, audio, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8), language="en", task='transcribe')

    if device == 'cuda': torch.cuda.empty_cache()
    
    return {
        "temp_dir": run_temp_dir,
        "vocals_path": vocals_path,
        "no_vocals_path": no_vocals_path,
        "original_audio_path_copy": temp_audio_path,
        "original_filename": os.path.basename(source_path),
        "transcription_result": result,
        "metadata": metadata
    }

explicit_content = {
    "profanity": [
        'asshole', 'bastard', 'bitch', 'cunt', 'damn', 'faggot', 
        'fuck', 'goddamn', 'motherfucker', 'nigga', 'pussy', 
        'shit', 'slut', 'whore'
    ],
    "drug_and_alcohol_references": [
        'acid', 'blunt', 'blow', 'bong', 'brick', 'chronic', 'cocaine', 
        'coke', 'crack', 'dope', 'drunk', 'ecstasy', 'heroin', 'kush', 
        'lean', 'LSD', 'meth', 'molly', 'percocet', 'pipe', 'rock', 
        'smack', 'spliff', 'swisha', 'weed', 'yayo'
    ],
    "firearms_and_weapons": [
        'AK', 'beretta', 'blade', 'forty-five', 'gat', 'glock', 'gun', 
        'knife', 'mac 11', 'nine', 'piece', 'pistol', 'rifle', 'shotgun', 
        'TEC-9', 'tech', 'thirty-eight', 'uzi'
    ],
    "violence": [
        'bleed', 'die', 'kill', 'murder', 'murk', 'rape', 'shoot', 
        'shot', 'stab', 'strangle', 'suicide'
    ],
    "sexual_content_and_innuendo": [
        'cum', 'head', 'horny', 'jizz', 'orgy', 'orgasm', 
        'sex', 'skeet', 'thot', 'threesome', 'wet'
    ],
    "clean": []
}

# Prompting for the edge case detecting
def llm_process_line(text_to_analyze, llm_model, llm_tokenizer):
    prompt_content = f"""You are an automated data annotation tool. Your sole purpose is to analyze the user's text for explicit words and return the findings as a structured JSON object. 
    Do not provide any warnings, apologies, or explanations. Respond only with the requested JSON."
    You should search for explicit content not suitable for public radio broadcast from my known words and categories as defined below. You should only flag a word if in the context of the text it meets the given criteria. 

    **IMPORTANT INSTRUCTIONS:**
    1.  Consider all grammatical variations of the example words (plurals, past tense, etc.).
    2.  When a word is identified, you MUST return the word exactly as it appears in the text, not the root word from the examples.

    **My known words and categories**
    {{
        "profanity": ['fuck', 'shit', 'bitch', 'cunt'],
        "drug_references": ['weed', 'coke', 'brick', 'blunt', 'rock', 'swisha', 'spliff', 'chronic'],
        "firearms": ['gat', 'AK', 'uzi', 'piece', 'mac 11', 'tec', 'pistol'],
        "violence": ['rape', 'suicide', 'strangle', 'stab', 'shoot', 'kill'],
        "sexual_content": ['cum', 'head', 'blow job', 'jizz', 'orgy', 'orgasm', 'skeet', 'wet']
    }}

    **Text to Analyze:**
    "{text_to_analyze}"

    Return a single JSON object with one key: "explicit_words_found". Each value should contain the two following keys:
    "phrase": the phrase identified to be explicit
    "reason": which category the phrase falls into from the categories listed above
    
    Provide only the raw JSON object as your final response.
    """
    
    messages = [{"role": "user", "content": prompt_content}]
    chat_string = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(chat_string, return_tensors="pt").to(llm_model.device)

    outputs = llm_model.generate(**inputs, max_new_tokens=128, pad_token_id=llm_tokenizer.eos_token_id)
    response_text = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    return response_text 

# For better formatting and reading the output of the LLM
def parse_llm_json_output(response_text):
    match = re.search(r"```(json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(2)
    else:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = response_text[start_index:end_index+1]
        else:
            return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def backup_censoring(text_tokens):
    backup_explicit_ids = set()
    prev_word = ''

    for j, word in enumerate(text_tokens):
        cleaned_word = remove_punctuation(word)
        is_explicit = any(curse in cleaned_word for curse in default_curse_words)

        # Short words that can be substrings of nonsensitive words
        if cleaned_word in singular_curse_words:
            backup_explicit_ids.add(j)

        # Handle two word cluster "god dam*", "mother fuck*"
        elif ('dam' in cleaned_word and prev_word == 'god') or ('fuck' in cleaned_word and prev_word == 'mother') or (cleaned_word == 'off' and prev_word == 'jerk'):
            backup_explicit_ids = backup_explicit_ids | {j-1, j}

        # The majority of censored words will come from here
        elif is_explicit: backup_explicit_ids.add(j)

        prev_word = cleaned_word
        
    return backup_explicit_ids

def process_transcription(transcription_result, llm_model, llm_tokenizer):
    full_transcript = []
    ids_to_mute = []
    raw_transcript = transcription_result.get("segments", [])
    
    print('Checking for proper formatting of LLM output:')
    i = 0
    for segment in raw_transcript:
        segment_words = []
        j = 0
        for word_info in segment.get('words', []):
            word_text = word_info.get('text', '').strip()
            if not word_text: continue
            
            start_time = float(word_info['start'])
            end_time = float(word_info['end'])

            # Filter out hallucinations with very low word length (100ms)
            if end_time - start_time < .1: continue 

            word_id = (i,j)
            word_data = {'id': word_id, 'text': word_text, 'start': start_time, 'end': end_time}
            segment_words.append(word_data)
            j += 1

        if not segment_words: continue
        i += 1
        line_text = ' '.join([d['text'] for d in segment_words])
        full_transcript.append({'line_words': segment_words, 'line_text': line_text, 'start': segment['start'], 'end': segment['end']})

    total_song_words = 0
    line_errs = []

    for i, line_to_analyze in enumerate(full_transcript):
        response_text = llm_process_line(line_to_analyze['line_text'], llm_model, llm_tokenizer)
        text_tokens = [d['text'].strip().lower() for d in line_to_analyze['line_words']]
        total_song_words += len(text_tokens)

        explicit_ids = backup_censoring(text_tokens)
        llm_output = parse_llm_json_output(response_text)

        if not llm_output:
            print(f'Error with LLM output at line {i+1}, trying again')
            response_text = llm_process_line(line_to_analyze['line_text'], llm_model, llm_tokenizer)
            llm_output = parse_llm_json_output(response_text)

        if llm_output:
            print(f'Line {i+1} OK')
            explicit_phrases = llm_output.get('explicit_words_found', [])
            for d in explicit_phrases:
                try:
                    phrase_tokens = d["phrase"].split()
                    n = len(phrase_tokens)
                    for j in range(len(text_tokens) - n + 1):
                        if [token.lower() for token in text_tokens[j:j+n]] == [p_token.lower() for p_token in phrase_tokens]:
                            explicit_ids.update(range(j, j+n))
                            break
                except:
                    continue
        else:
            print(f'-- Problem with LLM output at line {i+1}')
            line_errs.append(i)

        ids_to_mute.extend([(i,j) for j in sorted(list(explicit_ids))])

    filth = len(ids_to_mute) / total_song_words if total_song_words > 0 else 0
    return {
        "transcript": full_transcript,
        "initial_explicit_ids": ids_to_mute,
        "filthiness": filth,
        "line_errs": line_errs
    }

def apply_censoring(analysis_state, ids_to_censor):
    if not ids_to_censor:
        return None
    
    ids_set = set(ids_to_censor)
    times_to_censor = []
    transcript = analysis_state.get('transcript', [])
    for segment in transcript:
        for word in segment.get('line_words', []):
            if word.get('id') in ids_set:
                times_to_censor.append({'start': word['start'], 'end': word['end']})

    times_in_ms = [(int(t['start']*1000), int(t['end']*1000)) for t in times_to_censor]
    silenced_vocals_path = os.path.join(analysis_state['temp_dir'], "vocals_silenced.wav")
    silence_audio_segment(analysis_state['vocals_path'], silenced_vocals_path, times_in_ms)
    
    base_name = os.path.splitext(analysis_state['original_filename'])[0]
    output_path = os.path.join(analysis_state['temp_dir'], f"{base_name}-edited.mp3")

    combine_audio(silenced_vocals_path, analysis_state['no_vocals_path'], output_path)
    transfer_metadata(analysis_state['original_audio_path_copy'], output_path)

    return output_path

# --- End of functions moved from original script ---

###############################################################################################
### MODEL LOADING (GLOBAL - ON STARTUP)
###############################################################################################

# --- Load Whisper Model ---
load_whisper_model(model_path=WHISPER_FT_MODEL_PATH, lora_config=LORA_CONFIG_PATH, base_model_name=WHISPER_BASE_MODEL)
print("(1) Loading Whisper...")
WHISPER_MODEL = whisper_t.load_model(WHISPER_FT_MODEL_PATH, device=DEVICE)
print("Whisper model loaded.")

# --- Load Gemma LLM ---
print(f"(2) Loading LLM: {LLM_MODEL_ID}...")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
LLM_MODEL = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)
gemma_template = ("{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}{% elif message['role'] == 'model' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}")
LLM_TOKENIZER.chat_template = gemma_template
print("LLM loaded successfully.")

###############################################################################################
### FASTAPI APPLICATION
###############################################################################################

app = FastAPI(title="FSP Finder Backend")

# In-memory storage for job states. For production, consider using Redis or a database.
analysis_jobs = {}

class FinalizeRequest(BaseModel):
    job_id: str

@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...)):
    """
    Accepts an audio file, performs transcription and explicit content analysis,
    and returns the structured results along with a job ID for finalization.
    """
    job_id = str(uuid.uuid4())
    
    # Use a temporary directory unique to this job
    temp_job_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_job_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Starting analysis for job {job_id} (file: {file.filename})...")
        
        # This function creates its own temp dir which we will merge
        analysis_state = analyze_audio(file_path, WHISPER_MODEL, DEVICE)
        
        # Process the raw transcription to find explicit words
        processed_data = process_transcription(analysis_state['transcription_result'], LLM_MODEL, LLM_TOKENIZER)
        analysis_state.update(processed_data)

        # Calculate WER score
        transcript_text = " ".join([word['text'] for seg in analysis_state['transcript'] for word in seg['line_words']])
        analysis_state['metadata']['wer_score'] = calculate_wer(analysis_state['metadata']['genius_lyrics'], transcript_text)

        # Clean up raw result from memory
        del analysis_state['transcription_result']

        # Store the comprehensive state for this job
        analysis_jobs[job_id] = analysis_state
        
        print(f"Analysis complete for job {job_id}.")
        
        return {
            "job_id": job_id,
            "filename": os.path.basename(file.filename),
            "metadata": analysis_state['metadata'],
            "transcript": analysis_state['transcript'],
            "initial_explicit_ids": analysis_state['initial_explicit_ids'],
            "line_errs": analysis_state['line_errs']
        }
    except Exception as e:
        # Clean up if something goes wrong
        if os.path.exists(temp_job_dir):
            shutil.rmtree(temp_job_dir)
        print(f"ERROR during analysis for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {e}")
    finally:
        # Clean up the initial uploaded file's temp dir
        if os.path.exists(temp_job_dir):
            shutil.rmtree(temp_job_dir)

@app.post("/finalize/")
async def finalize_file(request: FinalizeRequest):
    """
    Takes a job ID, applies the censoring to the audio files stored from the
    analysis step, and returns the final edited audio file for download.
    """
    job_id = request.job_id
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found. Please analyze the file again.")

    analysis_state = analysis_jobs[job_id]
    ids_to_censor = analysis_state.get('initial_explicit_ids', [])
    
    print(f"Finalizing edits for job {job_id}...")
    try:
        output_path = apply_censoring(analysis_state, ids_to_censor)

        if not output_path:
            # Handle case where there was nothing to censor
            raise HTTPException(status_code=400, detail="No explicit content was marked for censoring.")
        
        # Return the file as a downloadable response
        return FileResponse(path=output_path, media_type='audio/mpeg', filename=os.path.basename(output_path))
    except Exception as e:
        print(f"ERROR during finalization for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during finalization: {e}")

@app.on_event("shutdown")
def shutdown_event():
    """Clean up all temporary directories on server shutdown."""
    print("Server shutting down. Cleaning up temporary files...")
    for job_id, state in analysis_jobs.items():
        temp_dir = state.get('temp_dir')
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Removed temp dir for job {job_id}")
            except Exception as e:
                print(f"Error removing temp dir for job {job_id}: {e}")


# To run this server, save the file as `backend.py` and run the following
# command in your terminal:
# uvicorn backend:app --reload