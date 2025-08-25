import whisper_timestamped as whisper_t
import whisper
import torch
import os
import demucs.separate
import re
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
import lyricsgenius
import jiwer
import shutil
import tempfile
import json



## Get a genius API key at https://genius.com/api-clients
## put your key in system environment at GENIUS_API_TOKEN or set it manually here
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN") 
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True)


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