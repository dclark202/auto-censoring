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

import whisper
import torch
import pandas as pd
import os
import demucs.separate
import re
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3

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



###################################################################################################################
# Primary function
###################################################################################################################

def process_audio(audio_path, model, device, delete_splits=True, save_log=True, additional_curses=set(), additional_times=[]):

    print('----------\n')

    output_dir = os.path.abspath('./edited')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

        print(f'- Processing audio at {audio_path}')
    
    full_audio_path = os.path.abspath(audio_path)
    audio_directory = os.path.dirname(audio_path)

    file_name, file_ext = os.path.splitext(os.path.basename(full_audio_path))

    temp_audio_path = os.path.abspath(os.path.join(audio_directory, 'song_to_edit' + file_ext))

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
    
    os.rename(full_audio_path, temp_audio_path)

    # mmake sure an appropriate audio file has been given
    if file_ext not in {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff'}:
        print(f'(!) Improper file type ({file_ext}), exiting')
        return []
    
    # demucs is particular about path formatting
    demucs_path = os.path.abspath(f"separated/mdx_extra/song_to_edit")
    vocals_path = os.path.join(demucs_path, "vocals.wav")
    no_vocals_path = os.path.join(demucs_path, "no_vocals.wav")

    if os.path.isfile(vocals_path) and os.path.isfile(no_vocals_path):
        print(f"- Isolated vocals and instruments tracks already exist ({demucs_path})")

    else:
        # separate vocals and instruments
        print("- Loading demucs. This will separate the vocals and instrument stems\n")
        demucs.separate.main(["--two-stems", "vocals", "-n", "mdx_extra", temp_audio_path])
    
    print(f'\n- Transcribing track with Whisper on {device} (note: you may get Triton kernel issues)')

    
    result = model.transcribe(vocals_path, 
                                language='en',
                                task='transcribe',
                                word_timestamps=True
                                )

    all_dicts = []
    full_transcript = []

    # primary transcription loop. Record each word with timestamps and segment level lyrics chunks
    for segment in result["segments"]:
        
        segment_transcript = []
        
        for word_info in segment['words']:
            temp_dict = dict()
            word = word_info['word'].strip()
            
            segment_transcript.append(word)

            temp_dict['word'] = remove_punctuation(word)
            temp_dict['start'] = float(word_info['start'])
            temp_dict['end'] = float(word_info['end'])
            temp_dict['prob'] = word_info['probability']
            
            all_dicts.append(temp_dict)
        
        full_transcript.append({'line': ' '.join(segment_transcript), 
                                'start': segment['start'], 
                                'end': segment['end']})

    transcript = {'lines': full_transcript, 'word': all_dicts}

    explicit = detect_explicit_content(transcript['word'], additional_curses=additional_curses)

    times = []

    for word_info in explicit:
        start = int(word_info['start']*1000)
        end = int(word_info['end']*1000)

        times.append((start, end))

    #times = sorted(list(set(times.extend(additional_times))))

    if not times:
        print('\n-- No explicit content found --')
        
        res ={'output_path': None, 'full_transcript': transcript['lines'], 'explicit_log': []}
        
        return res

    print(f'- Explicit content found ({len(times)} {'item' if len(times) == 1 else 'items'}). Applying edits.')

    silence_audio_segment(vocals_path, vocals_path, times)

    output_path = os.path.join(output_dir, f"{file_name}-edited.mp3")

    ## Combine audio
    combine_audio(vocals_path, no_vocals_path, output_path)

    ## Transfer metadata to the new file
    transfer_metadata(original_audio_path=temp_audio_path, edited_audio_path=output_path)

    for word_info in explicit:
        word_info['start'] = seconds_to_minutes(word_info['start'])
        word_info['end'] = seconds_to_minutes(word_info['end'])
        word_info['prob'] = f'{round(float(word_info['prob'])*100, 2)}%'

    res = {'output_path': output_path, 'full_transcript': transcript['lines'], 'explicit_log': explicit}
    
    print('----------')
    print('Process finished')
    print(f'- Edited audio file saved in {output_dir}')

    if device == 'cuda':
        torch.cuda.empty_cache()

    os.remove(temp_audio_path)
    os.remove(vocals_path)
    os.remove(no_vocals_path)
    os.rmdir(demucs_path)
    return res

def process_audio_batch(audio_paths, model_name='medium.en', delete_splits=True, save_log=True, additional_curses=set()):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading whisper ({model_name}) on {device}...')

    model = whisper.load_model(model_name, device=device)
    
    print('----------\n')

    output_dir = os.path.abspath('./edited')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    res = []

    for i, audio_path in enumerate(audio_paths):
        print(f'Track {i+1} -- Processing audio at {audio_path}')
        
        full_audio_path = os.path.abspath(audio_path)
        file_name, file_ext = os.path.splitext(os.path.basename(full_audio_path))

        # mmake sure an appropriate audio file has been given
        if file_ext not in {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aiff'}:
            print(f'(!) Improper file type ({file_ext}), exiting')
            return []
        
        # demucs is particular about path formatting
        demucs_path = os.path.abspath(f"separated/mdx_extra/{file_name}")
        vocals_path = os.path.join(demucs_path, "vocals.wav")
        no_vocals_path = os.path.join(demucs_path, "no_vocals.wav")

        if os.path.isfile(vocals_path) and os.path.isfile(no_vocals_path):
            print(f"- Isolated vocals and instruments tracks already exist ({demucs_path})")

        else:
            # separate vocals and instruments
            print("- Loading demucs. This will separate the vocals and instrument stems\n")
            demucs.separate.main(["--two-stems", "vocals", "-n", "mdx_extra", audio_path])
        
        print(f'\n- Transcribing track (note: you may get Triton kernel issues)')

        result = model.transcribe(audio_path, 
                                  language='en',
                                  task='transcribe',
                                  word_timestamps=True
                                  )

        all_dicts = []
        full_transcript = []

        # primary transcription loop. Record each word with timestamps and segment level lyrics chunks
        for segment in result["segments"]:
            
            segment_transcript = []
            
            for word_info in segment['words']:
                temp_dict = dict()
                word = word_info['word'].strip()
                
                segment_transcript.append(word)

                temp_dict['word'] = remove_punctuation(word)
                temp_dict['start'] = float(word_info['start'])
                temp_dict['end'] = float(word_info['end'])
                temp_dict['prob'] = word_info['probability']
                
                all_dicts.append(temp_dict)
            
            full_transcript.append({'line': ' '.join(segment_transcript), 
                                    'start': segment['start'], 
                                    'end': segment['end']})

        transcript = {'lines': full_transcript, 'word': all_dicts}

        explicit = detect_explicit_content(transcript['word'], additional_curses=additional_curses)

        times = []

        for word_info in explicit:
            start = int(word_info['start']*1000)
            end = int(word_info['end']*1000)

            times.append((start, end))

        if not times:
            print('\n-- No explicit content found --')
            
            res.append({'output_path': None, 
                        'full_transcript': transcript['lines'], 
                        'explicit_log': []})
            
            continue

        print(f'- Explicit content found ({len(times)} {'item' if len(times) == 1 else 'items'}). Applying edits.')

        silence_audio_segment(vocals_path, vocals_path, times)

        output_path = os.path.join(output_dir, f"{file_name}-edited.mp3")

        ## Combine audio
        combine_audio(vocals_path, no_vocals_path, output_path)

        if delete_splits:
            os.remove(vocals_path)
            os.remove(no_vocals_path)
            os.rmdir(demucs_path)

        ## Transfer metadata to the new file
        transfer_metadata(original_audio_path=full_audio_path, edited_audio_path=output_path)

        for word_info in explicit:
            word_info['start'] = seconds_to_minutes(word_info['start'])
            word_info['end'] = seconds_to_minutes(word_info['end'])
            word_info['prob'] = f'{round(float(word_info['prob'])*100, 2)}%'

        res.append({'output_path': output_path, 
                    'full_transcript': transcript['lines'], 
                    'explicit_log': explicit})
        
        print()

    print('----------')
    print('Process finished')
    print(f'- Edited audio files saved in {output_dir}')

    if device == 'cuda':
        torch.cuda.empty_cache()

    return res