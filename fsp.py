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

GENIUS_API_TOKEN = ""  # your token here
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True)

default_curse_words = {'fuck', 'shit', 'piss', 'bitch', 'nigg', 'cock', 'faggot', 'cunt', 'pussy', 'dick', 'whore', 'goddam'}

# --- Helper Functions (remove_punctuation, get_metadata, etc.) ---
def remove_punctuation(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower()

def silence_audio_segment(input_audio_path, output_audio_path, times):
    audio = AudioSegment.from_file(input_audio_path)
    for (start_ms, end_ms) in times:
        before_segment = audio[:start_ms]
        target_segment = audio[start_ms:end_ms] - 60
        after_segment = audio[end_ms:]
        audio = before_segment + target_segment + after_segment
    audio.export(output_audio_path, format='wav')

def combine_audio(path1, path2, outpath):
    audio1 = AudioSegment.from_file(path1, format='wav')
    audio2 = AudioSegment.from_file(path2, format='wav')
    combined_audio = audio1.overlay(audio2)
    combined_audio.export(outpath, format="mp3")

def get_metadata(original_audio_path):
    try:
        audio_orig = EasyID3(original_audio_path)
        metadata = {'title': audio_orig.get('title', [None])[0], 'artist': audio_orig.get('artist', [None])[0], 'album': audio_orig.get('album', [None])[0], 'year': audio_orig.get('date', [None])[0]}
    except Exception:
        metadata = {'title': 'N/A', 'artist': 'N/A', 'album': 'N/A', 'year': 'N/A'}
    return metadata

def transfer_metadata(original_audio_path, edited_audio_path):
    try:
        audio_orig = EasyID3(original_audio_path)
        audio_edit = EasyID3(edited_audio_path)
        for key in audio_orig.keys():
            audio_edit[key] = audio_orig[key]
        audio_edit.save()
    except Exception as e:
        print(f"Could not transfer metadata: {e}")

def seconds_to_minutes(time):
    mins = int(time // 60)
    secs = int(time % 60)
    return f"{mins}m {secs}s"

def get_genius_url(artist, song_title):
    if not artist or not song_title or artist == 'N/A' or song_title == 'N/A': return None
    try:
        song = genius.search_song(song_title, artist)
        return song.url if song else None
    except Exception: return None

def calculate_wer(ground_truth, hypothesis):
    if not ground_truth or not hypothesis or "not available" in ground_truth.lower(): return None
    try:
        transformation = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ExpandCommonEnglishContractions(), jiwer.RemoveEmptyStrings()])
        error = jiwer.mer(transformation(ground_truth), transformation(hypothesis))
        return f"{error:.3f}"
    except Exception: return "Error"

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
    Performs audio separation and transcription. Does NOT apply any edits.
    Returns a state dictionary with paths to temp files and the transcript.
    """
    if progress: progress(0, desc="Setting up temporary directory...")
    run_temp_dir = tempfile.mkdtemp()
    
    source_path = os.path.abspath(audio_path)
    
    # This line is changed to use the standardized filename 'temp_audio.mp3'
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
        word_key, prob_key = 'word', 'probability'
    else:
        audio = whisper_t.load_audio(vocals_path)
        result = whisper_t.transcribe(model, audio, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), language="en", task='transcribe')
        word_key, prob_key = 'text', 'confidence'

    full_transcript = []
    initial_explicit_times = []
    
    for segment in result["segments"]:
        segment_words = []
        for word_info in segment.get('words', []):
            word_text = word_info.get(word_key, '').strip()
            if not word_text: continue
            
            cleaned_word = remove_punctuation(word_text)
            is_explicit = any(curse in cleaned_word for curse in default_curse_words)
            
            start_time = float(word_info['start'])
            end_time = float(word_info['end'])
            
            word_data = {'text': word_text, 'start': start_time, 'end': end_time, 'prob': word_info[prob_key]}
            segment_words.append(word_data)
            
            if is_explicit:
                initial_explicit_times.append({'start': start_time, 'end': end_time})
        
        full_transcript.append({'line_words': segment_words, 'start': segment['start'], 'end': segment['end']})

    transcript_text = " ".join([word['text'] for seg in full_transcript for word in seg['line_words']])
    metadata['wer_score'] = calculate_wer(metadata['genius_lyrics'], transcript_text)

    if device == 'cuda': torch.cuda.empty_cache()
    
    return {
        "temp_dir": run_temp_dir,
        "vocals_path": vocals_path,
        "no_vocals_path": no_vocals_path,
        "original_audio_path_copy": temp_audio_path,
        "original_filename": os.path.basename(source_path),
        "transcript": full_transcript,
        "initial_explicit_times": initial_explicit_times,
        "metadata": metadata
    }

##############################################
# STEP 2: Apply Censoring and Finalize Audio #
##############################################

def apply_censoring(analysis_state, times_to_censor, progress=None):
    """
    Takes the state from analyze_audio and a final list of timestamps,
    applies silencing, and creates the final audio file.
    """
    if not times_to_censor:
        # If nothing to censor, just clean up and return nothing.
        shutil.rmtree(analysis_state['temp_dir'])
        return None
    
    if progress: progress(0, desc="Applying silence to vocal track...")
    times_in_ms = [(int(t['start']*1000), int(t['end']*1000)) for t in times_to_censor]
    silence_audio_segment(analysis_state['vocals_path'], analysis_state['vocals_path'], times_in_ms)

    output_dir = os.path.abspath('./edited')
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(analysis_state['original_filename'])[0]
    output_path = os.path.join(output_dir, f"{base_name}-edited.mp3")

    if progress: progress(0.6, desc="Combining audio tracks...")
    combine_audio(analysis_state['vocals_path'], analysis_state['no_vocals_path'], output_path)
    transfer_metadata(analysis_state['original_audio_path_copy'], output_path)

    # Clean up all temporary files
    shutil.rmtree(analysis_state['temp_dir'])

    return output_path