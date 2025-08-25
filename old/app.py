import gradio as gr
import os, re, html, json, shutil, tempfile
import torch
import whisper_timestamped as whisper_t
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import whisper
import demucs.separate
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
import lyricsgenius
import jiwer


###### Ideas ########
# - Javascript for toggling individual words to mute --> playright
# - Use LLM to determine what is "explicit" in the ouputs --> structured output?
# - Mute explicit nonvocal sounds: e.g., gun shots, sex scenes, etc.
# - Additional words to censor at the beginning screen ?
# - History


## Get a genius API key at https://genius.com/api-clients
## put your key in system environment at GENIUS_API_TOKEN or set it manually here
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN") 
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True)


# Print the start time
print(f"Executing {os.path.basename(__file__)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## 1. Load the Llama Model for edge case detection
llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# This is the magic that lets you run the model in <10GB VRAM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading {llama_model_id}...")
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically maps the model to your GPU
)
print('Done.')

################ Load models

## 1. Toxicity filter. Using the base version 
# print('Loading toxicity classifier...')
# tox_model = "cardiffnlp/twitter-roberta-large-sensitive-multilabel"
# tox_tokenizer = AutoTokenizer.from_pretrained(tox_model)
# tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model)
# tox_model.to(device)
# tox_pipe = pipeline("text-classification", model=tox_model, tokenizer=tox_tokenizer, device=device, top_k=2)



## 2. Create our Whisper model from the LoRA weights
## Whisper_timestamped requires the entire model to be saved, this saves static storage space by only saving the lora config
def load_whisper_model(model_path, lora_config, base_model_name="openai/whisper-medium.en"):
    # If the model exists already we're good to go
    if os.path.exists(f'./{model_path}/model.safetensors'):
        print(f'Fine tuned model at {model_path} already exists')
        return
    
    print(f'Fine-tuned model not found. Creating model from LoRA configuration at {lora_config}')
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(model, lora_config)

    model = model.merge_and_unload()
    model.save_pretrained(model_path, save_serialization=False)
    
    print(f'Whisper model from {lora_config} saved at {model_path}')
    return

# Where fsp.py expects to find our fine-tuned model
model_path = 'whisper-medium-ft'
lora_config = './lora_config'

load_whisper_model(model_path=model_path, lora_config=lora_config)

###### Helper functions #######

#############################################################################
### just a heads up there's a bunch of curse words and racial slurs below ###
#############################################################################


# List of words to search for to be muted:
# The way this works currently is that we look for these words as **substrings** of each transcribed word
# this means that 'fuck' handles all versions 'fucking', 'motherfucker', 'fucked', etc.
# This method is a bit crude as it can lead to some false positive, ex. 'Dickens' would be censored.
# Consider using an LLM on the output for classification?  
default_curse_words = {
    'fuck', 'shit', 'piss', 'bitch', 'nigg', 'dyke', 'cock', 'faggot', 
    'cunt', 'tits', 'pussy', 'dick', 'asshole', 'whore', 'goddam',
    'douche', 'chink', 'tranny', 'slut', 'jizz', 'kike', 'gook'
}

# Words for which the substring method will absolutely not work
singular_curse_words = {
    'fag', 'fags', 'faggy', 'cum', 'hell', 'spic', 'spics', 'clit', 
    'clits', 'wank', 'ass', 'asses', 'asswipe', 'asswipes', 'asscrack',
    'asscracks', 'wanks', 'cums', 'tit', 
}

always_bad = {'fuck', 'shit', 'cunt', 'bitch', 'pussy', 'nigg', 'goddam', 'faggot', 'asshole'}


######################################################
# Helper functions required for the gradio interface #
######################################################

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

# Probably overcomplicated function to convert time in seconds to mm:ss format
def seconds_to_minutes(time):
    mins = int(time // 60)
    secs = int(time % 60)

    if secs == 0:
        return f'{mins}:00'

    elif secs < 10:
        return f'{mins}:0{secs}'

    else:
        return f"{mins}:{secs}"

# Lookup url on genius of lyrics for given song
def get_genius_url(artist, song_title):
    if not artist or not song_title or artist == 'N/A' or song_title == 'N/A': return None
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
    if not artist or not song_title or artist == 'N/A' or song_title == 'N/A': return "Lyrics not available (missing metadata)."
    try:
        song = genius.search_song(song_title, artist)
        return song.lyrics if song else "Could not find lyrics on Genius."
    except Exception: return "An error occurred while searching for lyrics."

##########################################################
# STEP 1: Analyze Audio, Separate Tracks, and Transcribe #
##########################################################

def analyze_audio(audio_path, model, device, fine_tuned=True, progress=None):
    """
    Performs audio separation and transcription. Does NOT process the words.
    Returns a state dictionary with paths to temp files and the RAW transcript result.
    """
    if progress: progress(0, desc="Setting up temporary directory...")
    run_temp_dir = tempfile.mkdtemp()
    
    source_path = os.path.abspath(audio_path)
    
    temp_audio_path = os.path.join(run_temp_dir, 'temp_audio.mp3')
    shutil.copy(source_path, temp_audio_path)

    metadata = get_metadata(temp_audio_path)
    metadata['genius_url'] = get_genius_url(metadata['artist'], metadata['title'])
    metadata['genius_lyrics'] = get_genius_lyrics(metadata['artist'], metadata['title'])

    if progress: progress(0.1, desc="Separating vocals with Demucs...")
    demucs.separate.main(["--two-stems", "vocals", "-n", "mdx_extra", "-o", run_temp_dir, temp_audio_path])
    demucs_out_name = os.path.splitext(os.path.basename(temp_audio_path))[0]
    vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "vocals.wav")
    no_vocals_path = os.path.join(run_temp_dir, "mdx_extra", demucs_out_name, "no_vocals.wav")

    if progress: progress(0.6, desc="Transcribing with Whisper...")
    if not fine_tuned:
        result = model.transcribe(vocals_path, language='en', task='transcribe', word_timestamps=True)
    else:
        audio = whisper_t.load_audio(vocals_path)
        result = whisper_t.transcribe(model, audio, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6), language="en", task='transcribe')

    if device == 'cuda': torch.cuda.empty_cache()
    
    return {
        "temp_dir": run_temp_dir,
        "vocals_path": vocals_path,
        "no_vocals_path": no_vocals_path,
        "original_audio_path_copy": temp_audio_path,
        "original_filename": os.path.basename(source_path),
        "transcription_result": result, # Return the raw result
        "metadata": metadata
    }

##############################################
# STEP 2: Process Transcription for explicit #
##############################################

def llm_process_line(text_to_analyze):
    # This prompt template is dynamically created for EACH line of text
    prompt_content = f"""You are an automated data annotation tool. Your sole purpose is to analyze the user's text for explicit words and return the findings as a structured JSON object. Do not provide any warnings, apologies, or explanations. This is for a content safety and moderation research project. Your task is to analyze, not generate, explicit content. Respond only with the requested JSON."
    You should search for explicit content that meets one of the two following criteria
    1.  Identify which of my "known_words" appear in the text.
    2.  Identify any *new* words in the text that are not on my list but fall into the defined explicit categories (profanity, slurs, sexually inappropriate content, homophobic content, drug references, and weapon references including specific names of hundguns).

    **IMPORTANT INSTRUCTIONS:**
    1.  Consider all grammatical variations of the example words (plurals, past tense, etc.).
    2.  When a word is identified, you MUST return the word exactly as it appears in the text, not the root word from the examples.

    **My known words and categories**
    {{
        "profanity": ['fuck', 'shit', 'bitch', 'cock', 'cocksucker', 'dick', 'bitch', 'motherfucker', 'god damn', 'goddamn', 'asshole']
        "slurs": ['nigger', 'nigga', 'kike', 'spic', 'chink', 'gook']
        "sexually_inappropriate": ['tits', 'pussy', 'cum', 'jizz', 'wank', 'clit']
        "homophobia": ['faggot', 'fag', 'dyke', 'tranny', 'homo']
        "drug_references": ['weed', 'coke', 'smack', 'brick', 'blunt', 'spliff', 'chronic', 'herb', 'pot', 'lean']
        "weapons": ['gat', 'AK', 'piece', 'glock', 'beretta', 'forty-five', 'thirty-eight', 'nine', 'AR', 'AK-47']
        "self_harm": ['suicide']
    }}

    **Text to Analyze:**
    "{text_to_analyze}"

    Return a single JSON object with one key: "explicit_words_found". The value should be a list of all the explicit words you identified in the text. Provide only the raw JSON object as your final response.
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant that only returns valid JSON."},
        {"role": "user", "content": prompt_content},
    ]

    input_ids = llama_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llama_model.device)

    outputs = llama_model.generate(input_ids, max_new_tokens=128, pad_token_id=llama_tokenizer.eos_token_id)
    response_text = llama_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return response_text 

## Merge fsp and app into one .py file
## Make sure process_transcription works correctly and sends ids in the correct format 
### e.g., (i, j) can replace word_i_j


def process_transcription(transcription_result):
    """
    Takes the raw result from whisper and processes it to find explicit words.
    Returns a dictionary with the full transcript structure and explicit word IDs.
    """
    full_transcript = []
    ids_to_mute = []

    for i, segment in enumerate(transcription_result.get("segments", [])):
        segment_words = []
        
        j = 0
        for word_info in segment.get('words', []):
            word_text = word_info.get('text', '').strip()
            if not word_text: continue
            
            start_time = float(word_info['start'])
            end_time = float(word_info['end'])

            # Filter out hallucinations with very low word length. 
            # 100ms is a generous lower bound for minimum possible word length
            if end_time - start_time < .1: 
                continue 

            word_id = f"word_{i}_{j}"

            word_data = {'id': word_id, 'text': word_text, 'start': start_time, 'end': end_time}
            segment_words.append(word_data)

            j += 1

        line_text = ' '.join([d['text'] for d in segment_words])

        full_transcript.append({'line_words': segment_words, 'line_text': line_text, 'start': segment['start'], 'end': segment['end']})

    for i, line_to_analyze in enumerate(full_transcript):
        print(f'--- Line {i} ---')
        response_text = llm_process_line(line_to_analyze['line_text'])
        text_tokens = [d['text'].strip() for d in line_to_analyze['line_words']]
        total_song_words += len(text_tokens)

        # Store the word_ids of the explicit content
        explicit_ids = set()

        try: 
            llm_output = json.loads(response_text)

            explicit_phrases = llm_output.get('explicit_words_found', [])
            for phrase in explicit_phrases:
                phrase_tokens = phrase.split()
                n = len(phrase_tokens)
                
                for j in range(len(text_tokens) - n + 1):
                    if [token.lower() for token in text_tokens[j:j+n]] == [p_token.lower() for p_token in phrase_tokens]:
                        explicit_ids = explicit_ids | set([k for k in range(j, j+n)])
                        break

        except (json.JSONDecodeError, KeyError) as e:
            print('(!) Error with LLM output')

        # print('Text:', line_to_analyze['line_text'])
        # print('LLM output:', llm_output)

        # Grab any of the always bad ones not captured by the LLM
        for j, token in enumerate(text_tokens):
            if any(w in token for w in always_bad):
                explicit_ids.add(j)
        
        explicit_ids = sorted(list(explicit_ids))
        #print('Words to mute and indices:', [(line_to_analyze['line_words'][j]['text'], j) for j in explicit_ids])
        ids_to_mute.extend([(i,j) for j in explicit_ids])

    # # Handles different dictionary keys from whisper vs whisper_timestamped
    # word_key = 'text' if 'text' in transcription_result.get('segments', [{}])[0].get('words', [{}])[0] else 'word'
    # prob_key = 'confidence' if 'confidence' in transcription_result.get('segments', [{}])[0].get('words', [{}])[0] else 'probability'
    
    # prev_word = ''
    # prev_word_id = None

    # for i, segment in enumerate(transcription_result.get("segments", [])):
    #     segment_words = []
        
    #     j = 0
    #     for word_info in segment.get('words', []):
    #         word_text = word_info.get(word_key, '').strip()
    #         if not word_text: continue
            
    #         cleaned_word = remove_punctuation(word_text)
    #         is_explicit = any(curse in cleaned_word for curse in default_curse_words)
            
    #         start_time = float(word_info['start'])
    #         end_time = float(word_info['end'])

    #         # Filter out hallucinations with very low word length. 
    #         # 100ms is a generous lower bound for minimum possible word length
    #         if end_time - start_time < .1: 
    #             continue 

    #         word_id = f"word_{i}_{j}"
            
    #         word_data = {'id': word_id, 'text': word_text, 'start': start_time, 'end': end_time, 'prob': word_info.get(prob_key, 0.0)}
    #         segment_words.append(word_data)

    #         if cleaned_word in singular_curse_words:
    #             initial_explicit_ids.append(word_id)

    #         elif ('dam' in cleaned_word and 'god' in prev_word) or ('fuck' in cleaned_word and 'mother' in prev_word) or (cleaned_word == 'off' and 'jerk' in prev_word):
    #             if prev_word_id: initial_explicit_ids.append(prev_word_id)
    #             initial_explicit_ids.append(word_id)

    #         elif is_explicit:
    #             initial_explicit_ids.append(word_id)

    #         prev_word = cleaned_word
    #         prev_word_id = word_id
    #         j += 1
            
    #     full_transcript.append({'line_words': segment_words, 'start': segment['start'], 'end': segment['end']})

    return {
        "transcript": full_transcript,
        "initial_explicit_ids": ids_to_mute
    }


##############################################
# STEP 3: Apply Censoring and Finalize Audio #
##############################################

def apply_censoring(analysis_state, ids_to_censor, progress=None):
    """
    Takes the state from analyze_audio and a final list of word IDs,
    translates IDs to timestamps, applies silencing, and creates the final audio file.
    """
    if not ids_to_censor:
        return None
    
    if progress: progress(0, desc="Applying silence to vocal track...")

    # Translate word IDs to timestamps
    ids_set = set(ids_to_censor)
    times_to_censor = []
    transcript = analysis_state.get('transcript', [])
    for segment in transcript:
        for word in segment.get('line_words', []):
            if word.get('id') in ids_set:
                times_to_censor.append({'start': word['start'], 'end': word['end']})

    times_in_ms = [(int(t['start']*1000), int(t['end']*1000)) for t in times_to_censor]
    silence_audio_segment(analysis_state['vocals_path'], analysis_state['vocals_path'], times_in_ms)
    
    base_name = os.path.splitext(analysis_state['original_filename'])[0]
    output_path = os.path.join(analysis_state['temp_dir'], f"{base_name}-edited.mp3")

    if progress: progress(0.6, desc="Combining audio tracks...")
    combine_audio(analysis_state['vocals_path'], analysis_state['no_vocals_path'], output_path)
    transfer_metadata(analysis_state['original_audio_path_copy'], output_path)

    return output_path



# Metadata display for the full transcriptions. Includes genius link if possible
def format_metadata_header(filename, metadata, explicit_word_count):
    title, artist, album, year = metadata.get('title', 'N/A'), metadata.get('artist', 'N/A'), metadata.get('album', 'N/A'), metadata.get('year', 'N/A')
    genius_url, wer_score = metadata.get('genius_url'), metadata.get('wer_score')
    genius_link = f"|| **[View lyrics on Genius]({genius_url})**" if genius_url else ""
    wer_display = f"| similarity score = {wer_score} (lower is better)" if wer_score and genius_url else ""
    
    status_message = ""
    if explicit_word_count == 0:
        status_message = "\n\n**✅ No explicit content found in this track.**"
        
    return f"### Details for: *{filename}*\n**Artist:** {artist} | **Song:** {title} | **Album:** {album} ({year}) {genius_link} {wer_display}{status_message}"

def generate_static_transcript(transcript_data, initial_ids):
    ids_to_mute = set(initial_ids)
    table_header = "<table><thead><tr><th style='width: 125px;'>Time</th><th>Line transcript</th></thead><tbody>"
    table_rows = []

    # all_lines = [" ".join([word['text'] for word in segment.get('line_words', [])]) for segment in transcript_data]

    # explicit_results = []
    # if all_lines:
    #     pipeline_outputs = tox_pipe(all_lines)
        
    #     for line_result in pipeline_outputs:
    #         flags = []

    #         for d in line_result:
    #             label = d['label']
    #             score = d['score']

    #             if score < 0.3: continue
    #             elif label == 'confilctual' or label == 'selfharm': flags.append('violence')
    #             elif label == 'profanity': flags.append('curse')
    #             elif label == 'drugs': flags.append('drugs')
    #             elif label == 'sex': flags.append('sex')

    #         explicit_results.append(flags)  

    for i, segment in enumerate(transcript_data):
        start_time_str, end_time_str = seconds_to_minutes(segment.get('start')), seconds_to_minutes(segment.get('end'))
        
        # explicit_flag = ""
        # if explicit_results:
        #     for flags in explicit_results[i]:
        #         if 'violence' in flags: explicit_flag += '💥'
        #         if 'curse' in flags: explicit_flag += '🤬'
        #         if 'drugs' in flags: explicit_flag += '🚬'
        #         if 'sex' in flags: explicit_flag += '🔞'
        
        words_in_line = segment.get('line_words', [])
        formatted_words = []

        for word in words_in_line:
            word_id = word.get('id')

            if word_id in ids_to_mute:
                formatted_words.append(f"<s>{html.escape(word['text'])}</s>")

            else:
                formatted_words.append(html.escape(word["text"]))

        formatted_line = " ".join(formatted_words)
        table_rows.append(f"<tr><td>{start_time_str} - {end_time_str}</td><td>{formatted_line}</td></tr>")
        
    return table_header + "".join(table_rows) + "</tbody></table>"

# Execute the whisper model for transcription
def handle_batch_analysis(files, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload one or more audio files.")

    yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), None, None, None, None, None

    try:
        model, fine_tuned = (whisper_t.load_model(model_path, device=device), True)
    except Exception as e:
        raise gr.Error(f"Error loading fine-tuned Whisper model: {e}")

    all_results = {}
    num_files = len(files)
    for i, audio_file in enumerate(files):
        progress((i + 1) / num_files, desc=f"Analyzing File {i + 1} of {num_files}")
        filename = os.path.basename(audio_file.name)
        
        # MODIFIED: Restructured analysis flow
        # Step 1: Perform demucs and raw transcription
        analysis_state = analyze_audio(audio_file.name, model, device, fine_tuned, progress=None)

        # Step 2: Process the raw transcription to find explicit words
        processed_data = process_transcription(analysis_state['transcription_result'])
        analysis_state['transcript'] = processed_data['transcript']
        analysis_state['initial_explicit_ids'] = processed_data['initial_explicit_ids']

        # Step 3: Calculate WER score now that we have the processed transcript
        transcript_text = " ".join([word['text'] for seg in analysis_state['transcript'] for word in seg['line_words']])
        analysis_state['metadata']['wer_score'] = calculate_wer(analysis_state['metadata']['genius_lyrics'], transcript_text)

        # Clean up raw result from memory
        del analysis_state['transcription_result']

        all_results[filename] = analysis_state
        print(f"Transcription complete for: {filename} (file {i+1} of {num_files})")

    file_list = list(all_results.keys())
    first_file_results = all_results[file_list[0]]
    explicit_count_first_file = len(first_file_results['initial_explicit_ids'])
    header = format_metadata_header(file_list[0], first_file_results['metadata'], explicit_count_first_file)
    transcript_html = generate_static_transcript(first_file_results['transcript'], first_file_results['initial_explicit_ids'])
    
    any_explicit_content = any(len(res['initial_explicit_ids']) > 0 for res in all_results.values())
    if any_explicit_content:
        apply_button_update = gr.update(interactive=True, value="Apply all edits")
    else:
        apply_button_update = gr.update(interactive=False, value="No edits to make")

    yield (
        gr.update(visible=False), 
        gr.update(visible=True), 
        gr.update(visible=False), 
        all_results, 
        gr.update(choices=file_list, value=file_list[0]), 
        header, 
        transcript_html,
        apply_button_update
    )

# Selecting between different transcripts
def update_details_view(selected_filename, all_results):
    if not selected_filename or not all_results:
        return "", ""
    
    file_results = all_results[selected_filename]
    explicit_word_count = len(file_results['initial_explicit_ids'])
    header = format_metadata_header(selected_filename, file_results['metadata'], explicit_word_count)
    transcript_html = generate_static_transcript(file_results['transcript'], file_results['initial_explicit_ids'])
    return header, transcript_html

# Apply the edits to all songs
def handle_batch_finalization(all_results, progress=gr.Progress()):
    if not all_results:
        raise gr.Error("No active analysis session. Please process files first.")

    output_paths = []
    num_files = len(all_results)
    for i, (filename, analysis_state) in enumerate(all_results.items()):
        progress((i + 1) / num_files, desc=f"Applying edits {i + 1} of {num_files}")
        ids_to_censor = analysis_state.get('initial_explicit_ids', [])
        output_path = apply_censoring(analysis_state, ids_to_censor, progress=None)
        if output_path:
            output_paths.append(output_path)
            
    status_message = f"✅ **Success!** {len(output_paths)} of {len(all_results)} files have been censored."

    yield (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        status_message,
        output_paths,
        gr.update(visible=True),
        gr.update(visible=False)
    )

# Clear temp files and return to start
def return_to_start(all_results):
    """Cleans up all temporary directories and resets the UI to its initial state."""
    if all_results:
        for analysis_state in all_results.values():
            temp_dir_path = analysis_state.get('temp_dir')
            if temp_dir_path and os.path.exists(temp_dir_path):
                try:
                    shutil.rmtree(temp_dir_path)
                except Exception as e:
                    print(f"Error removing temporary directory {temp_dir_path}: {e}")
                    
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True, interactive=True),
        gr.update(choices=[], value=None, visible=True),
        None,
        "",
        "",
        "",
        None,
        None
    )


######  Gradio UI   ########

## CSS for formatting
css = """
#main-container { max-width: 1250px; margin: auto; }
#main-container .prose { font-size: 15px !important; }
#upload-view { max-width: 60%; margin: 0 auto; }
#loading-view { min-height: 500px; display: flex; justify-content: center; align-items: center; }
#apply-button { background-color: #3d9c3e !important; color: white !important; }
#processed-files-radio { min-height: 300px; }
s { color: #d32f2f; text-decoration: line-through; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="FSP Finder", css=css) as demo:
    analysis_results_state = gr.State(None)

    with gr.Column(elem_id="main-container"):
        gr.Markdown("# FSP Finder - AI-powered explicit content detector")
        gr.Markdown("Detects and automatically censors explicit content in music files. For source code and more details, visit our [github page](https://github.com/dclark202/auto-censoring).")
        gr.Markdown("---")

        with gr.Column(visible=True) as upload_view:
            gr.Markdown("### How to use")
            gr.Markdown('- Upload one or more audio files using the box below. Most common audio formats are accepted (e.g., `.mp3`, `.wav`, etc.).')
            gr.Markdown(f'- Click the **Process audio** button to create the transcriptions of the uploaded track(s). You will have a chance to review the edits before applying the censoring.')
        
            files_input = gr.File(label="Upload audio files", file_count="multiple", elem_id="upload-view", file_types=["audio"])
            process_button = gr.Button("Process audio", elem_id="upload-view")
        
            gr.Markdown('---')
            gr.Markdown('### How it works')
            gr.Markdown("This app uses a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to create a lyrics transcript of the uploaded music files. Explicit content (e.g., curse words) are then searched for in the lyrics transcript and highlighted. The vocals stem of the track is split off from the song using [demucs](https://github.com/facebookresearch/demucs) and muted at the appropriate times to create a high-quality edited version of the song.")

        with gr.Column(visible=False) as review_view:
            gr.Markdown("### Review transcript(s) and apply edits")
            gr.Markdown(f'Words to be censored will appear in <caption>{html.escape("red strikethrough")}</s> text in the transcript below. Apply edits by clicking **Apply all edits** below.')
            gr.Markdown("""Entries in the **Explicit flag** column are determined by running the corresponding line through a [toxicity filter](https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel). 
                         
                        - 💥 = violence or self harm
                        - 🤬 = curse words
                        - 🚬 = drugs
                        - 🔞 = sexual content
                        
                        We are currently working on allowing users to select additional words to censor from the full transcript, this flag should guide users towards identifying additional potentially explicit lines.""")
            gr.Markdown("**Note**: Whisper's processing is not deterministic and it can sometimes get confused and hallucinate with audio. If your transcription seems inaccurate (e.g., a line contains the same word repeated *many* times, or a line contains a significant amount of transcribed text not present in the song), please try running the program again on that song.")
            
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    processed_files_selector = gr.Radio(label="Select a file to view its transcript", interactive=True, elem_id="processed-files-radio")
                    apply_button = gr.Button("Apply all edits", elem_id="apply-button", interactive=False)
                    return_to_start_button = gr.Button("Return to start")
                    with gr.Column(visible=False) as final_view:
                        final_status_output = gr.Markdown()
                        edited_files_output = gr.File(label="Download your edited files", file_count="multiple")

                with gr.Column(scale=3):
                    details_header = gr.Markdown()
                    with gr.Accordion("Full audio transcript", open=True):
                        transcript_output = gr.HTML()

        with gr.Column(visible=False, elem_id="loading-view") as loading_view:
            gr.Markdown("## ⏳ Processing... please wait")

    process_button.click(
        fn=handle_batch_analysis,
        inputs=[files_input],
        outputs=[upload_view, review_view, loading_view, analysis_results_state, processed_files_selector, details_header, transcript_output, apply_button]
    )

    processed_files_selector.change(
        fn=update_details_view,
        inputs=[processed_files_selector, analysis_results_state],
        outputs=[details_header, transcript_output]
    )
    
    apply_button.click(
        fn=handle_batch_finalization,
        inputs=[analysis_results_state],
        outputs=[review_view, loading_view, final_view, final_status_output, edited_files_output, processed_files_selector, apply_button]
    )

    return_to_start_button.click(
        fn=return_to_start,
        inputs=[analysis_results_state],
        outputs=[
            upload_view,
            review_view,
            final_view,
            apply_button,
            processed_files_selector,
            analysis_results_state,
            details_header,
            transcript_output,
            final_status_output,
            edited_files_output,
            files_input
        ],
        js="() => { if (confirm('Are you sure you want to return to the start? All current analysis will be lost.')) { return true; } else { return false; } }"
    )

demo.launch(share=True, favicon_path='fav.png')