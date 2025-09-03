# backend.py
import os, re, json, shutil, tempfile, uuid
from datetime import datetime

# FastAPI and related imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

# ML libraries
import torch
import transformers
from transformers import AutoTokenizer, WhisperForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import whisper_timestamped as whisper
import demucs.separate
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# LangChain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import OutputFixingParser

# Audio processing
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
if not GENIUS_API_TOKEN: print("WARNING: GENIUS_API_TOKEN environment variable not set. Genius API features will fail.")

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

### JSON Parser ###

class LLMJsonResponse(BaseModel):
    explicit_phrases_found: List[str] = Field(description="A list of the explicit phrases found in the text")

def my_custom_json_parser(response_text: str):
    """
    Finds a JSON string in the LLM's response, then parses and validates it.
    """
    # This regex is a good way to find a JSON block
    match = re.search(r"```(json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(2)
    else:
        # Fallback to finding the first and last curly braces
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = response_text[start_index:end_index+1]
        else:
            print("❌ No JSON object found in the response.")
            # It's good practice to raise an exception here for the batch handler
            raise ValueError("No JSON object found")
    
    try:
        # We can use the Pydantic model directly to parse and validate
        validated_data = LLMJsonResponse.model_validate_json(json_str)
        # Return the data as a standard Python dictionary
        return validated_data.model_dump()
        
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"❌ Custom parser failed: {e}")
        # Raise an exception so the batch handler knows this item failed
        raise ValueError(f"JSON validation failed: {e}")

## cam delete ^^


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

# Removes all punctuation and returns lower case only words
def remove_punctuation(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower()

# For silencing the audio tracks at the indicated times
def silence_audio_segment(input_audio_path, output_audio_path, times):
    audio = AudioSegment.from_file(input_audio_path)
    for (start_ms, end_ms) in times:
        before_segment = audio[:start_ms]
        target_segment = audio[start_ms:end_ms] - 60
        after_segment = audio[end_ms:]
        audio = before_segment + target_segment + after_segment
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

    print()
    demucs.separate.main([
        "--two-stems", "vocals", # separate vocals from non-vocals
        "--shifts", "2", # two shifts. For better quality use more!
        "-j", "8", # num parallel jobs
        "-n", "mdx_extra", # mdx_extra transcription model
        "-o", run_temp_dir, # set the output directory
        temp_audio_path # input file
    ])
    
    demucs_out_name = os.path.splitext(os.path.basename(temp_audio_path))[0]
    vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "vocals.wav")
    no_vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "no_vocals.wav")

    audio = whisper.load_audio(vocals_path) # change to preprocessed_path to use exp settings

    print(f'\nTranscribing audio...')
    result = whisper.transcribe(
        model, 
        audio, 
        beam_size=5, 
        best_of=5, 
        remove_empty_words=True, # revomves nonsense from the end of the transcript
        #refine_whisper_precision=.8, # ??
        #vad='silero', # Use silero to remove sections of silence, 
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8), 
        language="en", 
        task='transcribe'
    )
    
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

def backup_censoring(text_tokens):
    backup_explicit_ids = set()
    prev_word = ''

    for j, word in enumerate(text_tokens):
        cleaned_word = remove_punctuation(word)
        is_explicit = any(curse in cleaned_word for curse in default_curse_words)

        # Short words that can be substrings of nonsensitive words
        if cleaned_word in singular_curse_words: backup_explicit_ids.add(j)

        # Handle two word cluster "god dam*", "mother fuck*"
        elif ('dam' in cleaned_word and prev_word == 'god') or ('fuck' in cleaned_word and prev_word == 'mother') or (cleaned_word == 'off' and prev_word == 'jerk'):
            backup_explicit_ids = backup_explicit_ids | {j-1, j}

        # The majority of censored words will come from here
        elif is_explicit: backup_explicit_ids.add(j)

        prev_word = cleaned_word
        
    return backup_explicit_ids

def process_transcription(transcription_result, llm_chain, use_llm=True):
    full_transcript = []
    ids_to_mute = []
    low_quality = []
    raw_transcript = transcription_result.get("segments", [])
    
    ## Create transcript
    i = 0
    for segment in raw_transcript:
        segment_words = []
        j = 0
        for word_info in segment.get('words', []):
            word_text = word_info.get('text', '').strip()
            if not word_text: continue
            
            start_time, end_time = float(word_info['start']), float(word_info['end'])

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

        if segment['avg_logprob'] < -1.0: low_quality.append(i+1)
            
    if low_quality: print(f'Low quality sections: {low_quality}')

    ## Use LLM for edge case detection
    line_errs = []
    
    if use_llm:
        lines_to_analyze = [{"text_to_analyze": line['line_text']} for line in full_transcript]

        try:
            print('\nCalling the LLM for explicit content detection...')
            llm_outputs = llm_chain.batch(lines_to_analyze, config={"max_concurrency": 8}) 

        except Exception as e:
            print(f"The LLM failed spectacularly")
            llm_outputs = [e] * len(lines_to_analyze)
        
        for i, llm_output in enumerate(llm_outputs):
            #print(llm_output)
            line_data = full_transcript[i]
            text_tokens = [d['text'].strip().lower() for d in line_data['line_words']]
            
            explicit_ids = backup_censoring(text_tokens)

            # If the JSON parsing for the LLM fails
            if isinstance(llm_output, Exception):
                print(f'Parsing error at line {i+1}')
                line_errs.append(i)
            
            else:
                explicit_phrases = llm_output.get('explicit_phrases_found', [])
                for phrase in explicit_phrases:
                    try:
                        phrase_tokens = phrase.split()
                        n = len(phrase_tokens)
                        for j in range(len(text_tokens) - n + 1):
                            if [token.lower() for token in text_tokens[j:j+n]] == [p_token.lower() for p_token in phrase_tokens]:
                                explicit_ids.update(range(j, j+n))
                        
                    except: continue

            ids_to_mute.extend([(i,j) for j in sorted(list(explicit_ids))])
    
    ## If not using the LLM, just do the backup censoring
    else:
        for line in full_transcript:
            text_tokens = [d['text'].strip().lower() for d in line['line_words']]
            explicit_ids = backup_censoring(text_tokens)
            ids_to_mute.extend([(i,j) for j in sorted(list(explicit_ids))])

    return {
        "transcript": full_transcript,
        "initial_explicit_ids": ids_to_mute,
        "line_errs": line_errs
    }

def apply_censoring(analysis_state, ids_to_censor):
    if not ids_to_censor: return None
    
    ids_set = {tuple(item) for item in ids_to_censor}
    times_to_censor = []
    transcript = analysis_state.get('transcript', [])

    for segment in transcript:
        for word in segment.get('line_words', []):
            if tuple(word.get('id')) in ids_set:
                times_to_censor.append({'start': word['start'], 'end': word['end']})

    times_in_ms = [(int(t['start']*1000), int(t['end']*1000)) for t in times_to_censor]
    silenced_vocals_path = os.path.join(analysis_state['temp_dir'], "vocals_silenced.wav")
    silence_audio_segment(analysis_state['vocals_path'], silenced_vocals_path, times_in_ms)
    
    base_name = os.path.splitext(analysis_state['original_filename'])[0]
    output_path = os.path.join(analysis_state['temp_dir'], f"{base_name}-edited.mp3")

    combine_audio(silenced_vocals_path, analysis_state['no_vocals_path'], output_path)
    transfer_metadata(analysis_state['original_audio_path_copy'], output_path)

    return output_path


###############################################################################################
### MODEL LOADING (GLOBAL - ON STARTUP)
###############################################################################################

# --- Load Whisper Model ---
load_whisper_model(model_path=WHISPER_FT_MODEL_PATH, lora_config=LORA_CONFIG_PATH, base_model_name=WHISPER_BASE_MODEL)
print("\n(1) Loading Whisper...")
WHISPER_MODEL = whisper.load_model(WHISPER_FT_MODEL_PATH, device=DEVICE)
print("Whisper model loaded.")

# --- Load Gemma LLM ---
print(f"\n(2) Loading LLM: {LLM_MODEL_ID}...")
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
LLM_MODEL = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)

gemma_template = ("{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}{% elif message['role'] == 'model' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}")
LLM_TOKENIZER.chat_template = gemma_template

# Create HF pipeline with custom tokenizer
pipe = pipeline(
    "text-generation",
    model=LLM_MODEL,
    tokenizer=LLM_TOKENIZER,
    max_new_tokens=512,
    device=None 
)
hf_pipeline = HuggingFacePipeline(pipeline=pipe)


prompt_str = """
You are an AI content moderator for a US radio station. Your goal is to identify words and phrases that would be considered 'indecent' or 'profane' under FCC broadcast standards, making them unsuitable for daytime airplay.

You must consider the context. Do not flag words that are merely references to mature themes (like drugs or weapons) unless they are used in a particularly graphic, gratuitous, or shocking manner. Differentiate between a mere mention and content designed to shock.

**Text to Analyze:**
"{text_to_analyze}"

Return a single JSON object with one key: "explicit_phrases_found" containing a list of each word or phrase EXACTLY as it appears in the text.

Provide only the raw JSON object as your final response.
"""

chat_prompt = ChatPromptTemplate.from_messages([("user", prompt_str)])

json_parser = JsonOutputParser()
fixing_parser = OutputFixingParser.from_llm(parser=json_parser, llm=hf_pipeline) # Fixes issues with the JSON parsing

# Read from left to right, not like functions >:( 
chain = chat_prompt | hf_pipeline | fixing_parser

###############################################################################################
### FASTAPI APPLICATION
###############################################################################################

app = FastAPI(title="FSP Finder Backend")

# In-memory storage for job states. For production, consider using Redis or a database.
analysis_jobs = {}

class FinalizeRequest(BaseModel):
    job_id: str
    ids_to_censor: Optional[List[List[int]]] = None # <-- ADD THIS LINE

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
        processed_data = process_transcription(analysis_state['transcription_result'], llm_chain=chain)
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
        if os.path.exists(temp_job_dir): shutil.rmtree(temp_job_dir)
        print(f"ERROR during analysis for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {e}")
    
    finally:
        # Clean up the initial uploaded file's temp dir
        if os.path.exists(temp_job_dir): shutil.rmtree(temp_job_dir)

@app.post("/finalize/")
async def finalize_file(request: FinalizeRequest):
    job_id = request.job_id
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    analysis_state = analysis_jobs[job_id]

    # FIX: This logic now correctly uses the user's edits if they exist.
    ids_to_censor = request.ids_to_censor
    if ids_to_censor is None:
        ids_to_censor = analysis_state.get('initial_explicit_ids', [])

    print(f"Finalizing edits for job {job_id} with {len(ids_to_censor)} words to censor...")
    try:
        # --- NEW CODE BLOCK TO SAVE TRAINING DATA ---
        # 1. Prepare the data object you want to save.
        data_to_save = {
            "metadata": analysis_state.get('metadata'),
            "transcript": analysis_state.get('transcript'),
            "final_censored_ids": ids_to_censor  # The user's final edits
        }
        
        # 2. Define the directory and create it if it doesn't exist.
        save_dir = "training_data"
        os.makedirs(save_dir, exist_ok=True)
        
        # 3. Create a unique filename for the JSON file.
        base_name = os.path.splitext(analysis_state['original_filename'])[0]
        json_filename = f"{base_name}_{job_id[:8]}.json"
        save_path = os.path.join(save_dir, json_filename)
        
        # 4. Write the data to the JSON file on your local machine.
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"Saved training data to {save_path}")
        # --- END OF NEW CODE BLOCK ---
        
        output_path = apply_censoring(analysis_state, ids_to_censor)

        if not output_path:
            raise HTTPException(status_code=400, detail="No explicit content was marked for censoring.")
        
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