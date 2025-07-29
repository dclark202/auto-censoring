"""
    process_audio is the main function here. 

    Inputs:
        - audio_path = path for the input file
        - whisper_type = 'medium.en' by default
        - delete_splits = deletes the vocals and instruments splits, True by default
        - additional_curses = Add more words you want to search for

    Outputs:
        - Transcription of lyrics w/ timestamps 
"""
import whisper_timestamped as whisper_t
import whisper
import torch
import pandas as pd
import os
import demucs.separate
import re
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
import lyricsgenius
import jiwer

GENIUS_API_TOKEN = "" # YOUR KEY HERE
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True)


default_curse_words = {'fuck', 'shit', 'piss', 'bitch', 'nigg', 'cock', 'faggot', 'cunt', 'pussy', 'dick', 'whore', 'goddam'}
## Use re to remove punctuation
def remove_punctuation(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower()

def remove_punctuation_extended(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = s.lower()
    while s[-1] == ' ':
        s = s[:-1]
    s = s.replace(' ', '_')
    return s

def detect_explicit_content(transcript, additional_curses=set()):
    curses = default_curse_words | additional_curses

    explicit = []

    for word_info in transcript:
        is_explicit = any(explicit_word in word_info['word'] for explicit_word in curses)

        if is_explicit: explicit.append(word_info)

    return explicit

## Applies silecning to input_audio_path at given list of times 
def silence_audio_segment(input_audio_path, output_audio_path, times):

    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)
    for (start_ms, end_ms) in times:
        # Select times to reverse
        before_segment = audio[:start_ms]

        # -60dB to the audio effectively mutes it
        target_segment = audio[start_ms:end_ms] - 60

        after_segment = audio[end_ms:]

        # Concatenate: this can be made faster, but it's not a priority
        audio = before_segment + target_segment + after_segment

    # Export the modified audio         
    audio.export(output_audio_path, format='wav') 

    return

# Combines two audio tracks via their paths (vocals and instruments for example)
# modifies audio in place
def combine_audio(path1, path2, outpath):
    audio1 = AudioSegment.from_file(path1, format='wav')
    audio2 = AudioSegment.from_file(path2, format='wav')

    combined_audio = audio1.overlay(audio2)

    # format='mp3' for mp3 files   
    combined_audio.export(outpath, format="mp3") 
    return

def get_metadata(original_audio_path):
    try:
        audio_orig = EasyID3(original_audio_path)

        metadata = dict()

        ## Add more if wanted
        metadata['title'] = audio_orig.get('title', [None])[0]
        metadata['artist'] = audio_orig.get('artist', [None])[0]
        metadata['album'] = audio_orig.get('album', [None])[0]
        metadata['year'] = audio_orig.get('date', [None])[0] 

    except Exception:
        metadata = {'title': 'N/A', 'artist': 'N/A', 'album': 'N/A', 'year': 'N/A'}

    return metadata

# Transfer the metadata from the original to the edited track
def transfer_metadata(original_audio_path, edited_audio_path):
    
    audio_orig = EasyID3(original_audio_path)
    audio_edit = EasyID3(edited_audio_path)

    metadata = dict()

    ## Add more if wanted
    metadata['title'] = audio_orig.get('title', [None])[0]
    metadata['artist'] = audio_orig.get('artist', [None])[0]
    metadata['album'] = audio_orig.get('album', [None])[0]
    metadata['date'] = audio_orig.get('date', [None])[0] # Often 'year' or full date
    metadata['tracknumber'] = audio_orig.get('tracknumber', [None])[0]
    
    # Apply metadata to edited track
    for key, value in metadata.items():
        audio_edit[key] = [str(value)]

    # and save
    audio_edit.save()
    return

# Useful for formatting
def seconds_to_minutes(time):
    mins = int(time // 60)
    secs = int(time % 60)
    return f"{mins}m {secs}s"

def get_genius_url(artist, song_title):
    """Searches Genius for a song and returns its URL."""
    if not artist or not song_title or artist == 'N/A' or song_title == 'N/A':
        return None
    try:
        song = genius.search_song(song_title, artist)
        return song.url if song else None
    except Exception as e:
        print(f"Error searching Genius: {e}")
        return None
    
def calculate_wer(ground_truth, hypothesis):
    """Calculates Word Error Rate and formats it as a percentage."""
    if not ground_truth or not hypothesis or "not available" in ground_truth.lower() or "could not find" in ground_truth.lower():
        return None
    try:
        # Basic pre-processing for a more accurate comparison
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings()
        ])
        
        processed_ground_truth = transformation(ground_truth)
        processed_hypothesis = transformation(hypothesis)
        
        error = jiwer.mer(processed_ground_truth, processed_hypothesis)
                          
        return f"{error:.3}" # Format as percentage string
    
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return "Error"
    
def get_genius_lyrics(artist, song_title):
    """Searches Genius for a song and returns its lyrics."""
    if not artist or not song_title or artist == 'N/A' or song_title == 'N/A':
        return "Lyrics not available (missing metadata)."
    
    try:
        song = genius.search_song(song_title, artist)
        return song.lyrics if song else "Could not find lyrics on Genius."
    
    except Exception as e:
        print(f"Error searching Genius: {e}")
        return "An error occurred while searching for lyrics."
    


###################################################################################################################
# Primary function
###################################################################################################################

def process_audio(audio_path, model, device, delete_splits=True, save_log=True, fine_tuned=True, additional_curses=set(), additional_times=[]):

    ## This handles the formatting of the input files
    output_dir = os.path.abspath('./edited')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    full_audio_path = os.path.abspath(audio_path)
    audio_directory = os.path.dirname(audio_path)

    file_name, file_ext = os.path.splitext(os.path.basename(full_audio_path))

    temp_audio_path = os.path.abspath(os.path.join(audio_directory, 'song_to_edit' + file_ext))

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    os.rename(full_audio_path, temp_audio_path)
    ###################################### 


    # make sure an appropriate audio file has been given
    if file_ext not in {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff'}:
        yield f'Error (!) Improper file type ({file_ext})'
        return 
    
    # Metadata for displaying
    metadata = get_metadata(temp_audio_path)
    metadata['genius_url'] = get_genius_url(metadata['artist'], metadata['title'])
    metadata['genius_lyrics'] = get_genius_lyrics(metadata['artist'], metadata['title'])

    # demucs is particular about path formatting
    demucs_path = os.path.abspath(f"separated/mdx_extra/song_to_edit")
    vocals_path = os.path.join(demucs_path, "vocals.wav")
    no_vocals_path = os.path.join(demucs_path, "no_vocals.wav")

    # separate vocals and instruments
    yield f"- Initializing demucs. This will separate the vocals and instrument stems"

    demucs.separate.main(["--two-stems", "vocals", "-n", "mdx_extra", temp_audio_path])
    
    yield f'- Transcribing track with Whisper'

    # Whisper and Whisper_timestamped have slightly different syntax
    if not fine_tuned:
        result = model.transcribe(vocals_path, 
                                    language='en',
                                    task='transcribe',
                                    word_timestamps=True)
        
        word_key = 'word'
        prob_key = 'probability'
        
    else:
        audio = whisper_t.load_audio(vocals_path)
        result = whisper_t.transcribe(model, 
                            audio, 
                            beam_size=5, 
                            best_of=5, 
                            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                            language="en",
                            task='transcribe')

        word_key = 'text'
        prob_key = 'confidence'

    all_dicts = []
    full_transcript = []

    # primary transcription loop. Record each word with timestamps and segment level lyrics chunks
    for segment in result["segments"]:
        
        # Temporary list to hold the words from the segment
        segment_transcript = []

        for word_info in segment['words']:
            temp_dict = dict()
            word = word_info[word_key].strip()
            
            segment_transcript.append(word)

            temp_dict['word'] = remove_punctuation(word)
            temp_dict['start'] = float(word_info['start'])
            temp_dict['end'] = float(word_info['end'])
            temp_dict['prob'] = word_info[prob_key]
            
            all_dicts.append(temp_dict)
        
        # Full audio transcript broken down by lines as identified in Whisper
        full_transcript.append({'line': ' '.join(segment_transcript), 
                                'start': segment['start'], 
                                'end': segment['end']})

    # Contains both word and line level transcripts
    transcript = {'lines': full_transcript, 'word': all_dicts}

    # Calculate WER between the full transcript and the genius lyrics
    transcript_text = " ".join([seg['line'] for seg in transcript['lines']])
    metadata['wer_score'] = calculate_wer(metadata['genius_lyrics'], transcript_text)

    ## Determine which word entries are explicit
    explicit = detect_explicit_content(transcript['word'], additional_curses=additional_curses)

    ## Create list of times to mute
    times = []
    for word_info in explicit:
        start = int(word_info['start']*1000)
        end = int(word_info['end']*1000)

        times.append((start, end))

    # If no explicit content is found we can exit early
    if not times:
        yield '\n-- No explicit content found --'
        res ={'output_path': None, 'full_transcript': transcript['lines'], 'explicit_log': []}
        yield res

    ## Else print how many have been found
    yield f'- Explicit content found ({len(times)} {'item' if len(times) == 1 else 'items'}). Applying edits.'

    ## Silence at the appropriate times
    silence_audio_segment(vocals_path, vocals_path, times)

    ## Define the output path
    output_path = os.path.join(output_dir, f"{file_name}-edited.mp3")

    ## Combine audio
    combine_audio(vocals_path, no_vocals_path, output_path)

    ## Transfer metadata to the new file
    transfer_metadata(original_audio_path=temp_audio_path, edited_audio_path=output_path)

    ## Proper formatting as Xm Ys for easier readability
    for word_info in explicit:
        word_info['start'] = seconds_to_minutes(word_info['start'])
        word_info['end'] = seconds_to_minutes(word_info['end'])
        word_info['prob'] = f'{round(float(word_info['prob'])*100, 2)}%'

    res = {'output_path': output_path, 
           'full_transcript': transcript['lines'], 
           'explicit_log': explicit,
           'metadata': metadata}
    
    yield '----------\nProcess finished'
    yield f'- Please download your edited files'

    ## Empty cache for efficient memory usage
    if device == 'cuda':
        torch.cuda.empty_cache()

    ## Remove the temp files
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    if os.path.exists(vocals_path):
        os.remove(vocals_path)

    if os.path.exists(no_vocals_path):
        os.remove(no_vocals_path)

    if os.path.exists(demucs_path):
        try:
            os.rmdir(demucs_path)
        except OSError:
            # The directory may not be empty if other processes are using it
            pass

    yield res