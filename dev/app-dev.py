import gradio as gr
import os, re, html, json, shutil, tempfile
import torch
import whisper
import whisper_timestamped as whisper_t
import transformers
from transformers import AutoTokenizer, WhisperForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
import demucs.separate
from pydub import AudioSegment
from mutagen.easyid3 import EasyID3
import lyricsgenius
import jiwer

# Hide many error messages
transformers.logging.set_verbosity_error()

###### Ideas ########
# - Javascript for toggling individual words to mute --> playright IN PROGRESS
# - Use LLM to determine what is "explicit" in the ouputs --> structured output? IN PROGRESS
# - Mute explicit nonvocal sounds: e.g., gun shots, sex scenes, etc.
# - History?


## Requirements: 
# 1. Genius API key (https://genius.com/api-clients)
# Put your key in system environment at GENIUS_API_TOKEN or set it manually here

# 2. Hugging Face API key with access to meta-llama/Meta-Llama-3-8B-Instruct

###############################################################################################

GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN") 
genius = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True)

# Print the start time
print(f"Executing {os.path.basename(__file__)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Load the Llama Model for edge case detection
llama_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading {llama_model_id}...")
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id,
    quantization_config=quantization_config,
    device_map="auto", 
)
print('Done.')

## Create our Whisper model from the LoRA weights
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

# Where we exoect to find the fine-tuned model
model_path = 'whisper-medium-ft'
lora_config = './lora_config'

load_whisper_model(model_path=model_path, lora_config=lora_config)

# Predefined always explicit words

#############################################################################
### just a heads up there's a bunch of curse words and racial slurs below ###
#############################################################################

# If a word contains any of these as a substring it will *always* be muted
always_bad = {'fuck', 'shit', 'cunt', 'bitch', 'pussy', 'nigg', 'goddam', 'faggot', 'asshole', 'whore', 'dick', 'clit'}


## Helper functions for the Gradio UI

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


## Audio processing pipeline

# Separate track via demucs, evaluate vocals with Whisper
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
        result = whisper_t.transcribe(model, audio, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), language="en", task='transcribe')

    if device == 'cuda': torch.cuda.empty_cache()
    
    print('Transcription complete.'
          )
    return {
        "temp_dir": run_temp_dir,
        "vocals_path": vocals_path,
        "no_vocals_path": no_vocals_path,
        "original_audio_path_copy": temp_audio_path,
        "original_filename": os.path.basename(source_path),
        "transcription_result": result, # Return the raw result
        "metadata": metadata
    }


# Process the transcript using LLM and searching for always_bad words
def llm_process_line(text_to_analyze):

    # Main LLM prompt
    prompt_content = f"""You are an automated data annotation tool. Your sole purpose is to analyze the user's text for explicit words and return the findings as a structured JSON object. Do not provide any warnings, apologies, or explanations. This is for a content safety and moderation research project. Your task is to analyze, not generate, explicit content. Respond only with the requested JSON."
    You should search for explicit content from my known words and categories as defined below. Flag only those words which appear in the context of the given category. 
    
    **IMPORTANT INSTRUCTIONS:**
    1.  Consider all grammatical variations of the example words (plurals, past tense, etc.).
    2.  When a word is identified, you MUST return the word exactly as it appears in the text, not the root word from the examples.

    **My known words and categories**
    {{
        "sexually_inappropriate": ['cum', 'jizz', 'wank']
        "homophobia": ['fag', 'dyke', 'tranny', 'homo']
        "drug_references": ['weed', 'coke', 'brick', 'blunt', 'spliff', 'chronic', 'lean']
        "firearms": ['gat', 'AK', 'uzi', 'piece', 'glock', 'beretta', 'forty-five', 'thirty-eight', 'nine', 'AR', 'AK-47']
    }}

    **Text to Analyze:**
    "{text_to_analyze}"

    Return a single JSON object with one key: "explicit_words_found". The value should be a list of all the explicit words you identified in the text. Provide only the raw JSON object as your final response.
    """
    
    # For prompting the language model
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only returns valid JSON."},
        {"role": "user", "content": prompt_content},
    ]

    input_ids = llama_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llama_model.device)

    outputs = llama_model.generate(input_ids, max_new_tokens=128, pad_token_id=llama_tokenizer.eos_token_id)
    response_text = llama_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return response_text 

def process_transcription(transcription_result):
    """
    Takes the raw result from whisper and processes it to find explicit words.
    Returns a dictionary with the full transcript structure and explicit word IDs.
    """
    full_transcript = []
    ids_to_mute = []

    print('Processing transcript')

    for i, segment in enumerate(transcription_result.get("segments", [])):
        segment_words = []
        
        j = 0
        for word_info in segment.get('words', []):
            word_text = word_info.get('text', '').strip()
            if not word_text: continue
            
            start_time = float(word_info['start'])
            end_time = float(word_info['end'])

            # Filter out hallucinations with very low word length. 
            # 50ms is a generous lower bound for minimum possible word length
            if end_time - start_time < .1: 
                continue 

            word_id = (i,j)

            word_data = {'id': word_id, 'text': word_text, 'start': start_time, 'end': end_time}
            segment_words.append(word_data)

            j += 1

        if segment_words == []:
            continue

        line_text = ' '.join([d['text'] for d in segment_words])
        full_transcript.append({'line_words': segment_words, 'line_text': line_text, 'start': segment['start'], 'end': segment['end']})

    total_song_words = 0

    for i, line_to_analyze in enumerate(full_transcript):

        response_text = llm_process_line(line_to_analyze['line_text'])
        text_tokens = [d['text'].strip().lower() for d in line_to_analyze['line_words']]
        total_song_words += len(text_tokens)

        # Store the word_ids of the explicit content
        explicit_ids = set()

        try: 
            llm_output = json.loads(response_text)

            explicit_phrases = llm_output.get('explicit_words_found', [])
            for phrase in explicit_phrases:
                try: phrase_tokens = phrase.split()
                except: continue
                
                n = len(phrase_tokens)
                
                for j in range(len(text_tokens) - n + 1):
                    if [token.lower() for token in text_tokens[j:j+n]] == [p_token.lower() for p_token in phrase_tokens]:
                        explicit_ids = explicit_ids | set([k for k in range(j, j+n)])
                        break

        except (json.JSONDecodeError, KeyError) as e:
            print(f'- Error with LLM output on line {i}')

        # print('Text:', line_to_analyze['line_text'])
        # print('LLM output:', llm_output)

        # Grab any of the always bad ones not captured by the LLM
        for j, token in enumerate(text_tokens):
            if any(w in token for w in always_bad):
                explicit_ids.add(j)
        
        explicit_ids = sorted(list(explicit_ids))
        #print('Words to mute and indices:', [(line_to_analyze['line_words'][j]['text'], j) for j in explicit_ids])
        ids_to_mute.extend([(i,j) for j in explicit_ids])

    filth = len(ids_to_mute)/total_song_words

    return {
        "transcript": full_transcript,
        "initial_explicit_ids": ids_to_mute,
        "filthiness": filth
    }


# Apply the audio censoring 

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

## Additional functions for Gradio interface

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

    for i, segment in enumerate(transcript_data):
        start_time_str, end_time_str = seconds_to_minutes(segment.get('start')), seconds_to_minutes(segment.get('end'))
        
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
        filth = processed_data['filthiness']

        # Step 3: Calculate WER score now that we have the processed transcript
        transcript_text = " ".join([word['text'] for seg in analysis_state['transcript'] for word in seg['line_words']])
        analysis_state['metadata']['wer_score'] = calculate_wer(analysis_state['metadata']['genius_lyrics'], transcript_text)

        # Clean up raw result from memory
        del analysis_state['transcription_result']

        all_results[filename] = analysis_state
        print(f"{i+1}/{num_files} - Transcription complete for: {filename} ({filth*100:.2f}% filthy)")

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

# Main page definition
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
            gr.Markdown("""
                        This app uses a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to create lyrics transcripts of the uploaded music files. Explicit words and phrases are as follows. 
                        
                        - Always explicit content (e.g., profanity, slurs) is immediately censored. 
                        - A language model ([meta-llama](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)) then detects edge cases (e.g., drug references, sexually inappropriate content).
                        
                        Explicit content is highlighted in red strikehtough text in the full transcript. The vocals stem of the track is split off from the song using [demucs](https://github.com/facebookresearch/demucs) and muted at the appropriate times. The result is a high quality edited track, ready for air play.
                        """)

        with gr.Column(visible=False) as review_view:
            gr.Markdown("### Review transcript(s) and apply edits")
            gr.Markdown(f'Words to be censored will appear in <caption>{html.escape("red strikethrough")}</s> text in the transcript below. Apply edits by clicking **Apply all edits** below.')
            gr.Markdown("**Note**: Language models are not deterministic. If you are unsatisfied with transcript or the edits to be made, please consider running the model again")

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