"""
    find_explicit is the main function here. 

    Inputs:
        - audio_path = path for the input file
        - whisper_type = 'large-v3-turbo' by default
        - delete_splits = deletes the vocals and instruments splits, True by default
        - save_log = saves a lot of the explicit content, True by default
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

## Use re to remove punctuation
def remove_punctuation(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower()

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

#
def seconds_to_minutes(time):
    mins = int(time // 60)
    secs = int(time % 60)
    return f"{mins}m {secs}s"

####
def find_explicit(audio_path, whisper_type='large-v3-turbo', delete_splits=True, save_log=True, additional_curses=set()):
    
    ## Define paths
    full_audio_path = os.path.abspath(audio_path)
    directory = os.path.dirname(full_audio_path)

    file_name, file_ext = os.path.splitext(os.path.basename(full_audio_path))

    print('Input file: ', full_audio_path)

    demucs_path = os.path.abspath(f"separated/mdx_extra/{file_name}")
    
    vocals_path = os.path.join(demucs_path, "vocals.wav")
    no_vocals_path = os.path.join(demucs_path, "no_vocals.wav")

    #### Load the model.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'\n<> Loading Whisper model {whisper_type} on {device} <>\n')
    model = whisper.load_model(whisper_type, device=device)

    # separate vocals and instruments
    print("<> Loading demucs <>\n")
    demucs.separate.main(["--two-stems", "vocals", "-n", "mdx_extra", audio_path])

    # Transcribe vocals
    print('\n<> Transcribing <>')
    print('Note: you may get Triton kernel issues\n')
    result = model.transcribe(vocals_path, word_timestamps=True)

    all = []
    for segment in result["segments"]:
        for word_info in segment['words']:

            word = word_info['word'].strip()
            word = remove_punctuation(word)
            
            start_time = float(word_info['start'])
            end_time = float(word_info['end'])
            prob = word_info['probability']
            
            all.append([word, start_time, end_time, prob])

    # Create Dataframe
    columns = ['word', 'start', 'end', 'prob']
    df_all = pd.DataFrame(all, columns=columns)

    # Use the additional_curses argument to add more
    curses = {'fuck', 'shit', 'bitch', 'nigga', 'nigger', 'cock', 'faggot', 'cunt', 'pussy', 'dick'} | additional_curses

    pattern = '|'.join(curses)

    # I noticed it can create duplicate entries for some reason, delete them then save the log to a .csv
    df_all = df_all.drop_duplicates()

    # Add a column which is boolean 1 for "is curse" and 0 for "is not curse"
    df_all['explicit'] = df_all['word'].str.contains(pattern, case=False, na=False, regex=True).astype(int)
    df = df_all[df_all['explicit']==1]

    if len(df) == 0:
        print("\n--No explicit content found--")
        return df_all

    else:
        print(f"\n(!) {len(df)} instances of explicit content found (!)\n")
        if save_log:
            csv_name = os.path.join(directory, f"{file_name}-explicit.csv")
            df.to_csv(csv_name, index=False)
    
    ## Create list of times to mute
    times = []
    for row in df.itertuples():
        #word = row[1]
        start = int(row[2]*1000)
        end = int(row[3]*1000)

        times.append((start, end))

    # Run the silencing script
    print('<> Applying silencing <>\n')
    silence_audio_segment(vocals_path, vocals_path, times)

    ## Output file name
    output_path = os.path.join(directory, f"{file_name}-edited.mp3")

    ## Combine audio
    print('<> Combining audio <>\n')
    combine_audio(vocals_path, no_vocals_path, output_path)

    if delete_splits:
        print('<> Deleting splits <>\n')
        os.remove(vocals_path)
        os.remove(no_vocals_path)
        os.rmdir(demucs_path)

    ## Transfer metadata
    transfer_metadata(original_audio_path=full_audio_path, edited_audio_path=output_path)
    
    ## Print the results
    print(f'Done')
    print(f' - Edited track output to {output_path}')
    
    if save_log:
        print(f' - Log saved to {csv_name}\n')

    print('Words removed:')
    print(df)

    return df_all