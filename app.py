import gradio as gr
import os
import torch
import whisper
import whisper_timestamped as whisper_t
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import WhisperForConditionalGeneration
from peft import PeftModel
import time
import re
import html
import json
import shutil # MODIFIED: Added shutil for directory cleanup
from fsp import analyze_audio, apply_censoring, default_curse_words, seconds_to_minutes

################ Load models


## 1. Toxicity filter. Using the base version 
print('Loading toxicity classifier...')
tox_model = "cardiffnlp/twitter-roberta-large-sensitive-multilabel"

tox_tokenizer = AutoTokenizer.from_pretrained(tox_model)
tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tox_model.to(device)
tox_pipe = pipeline("text-classification", model=tox_model, tokenizer=tox_tokenizer, device=device, top_k=2)

## 2. Create our Whisper model from the LoRA weights
## Whisper_timestamped requires the entire model to be saved, this saves static storage space by only saving the lora config
def load_whisper_model(model_path, lora_config, base_model_name="openai/whisper-medium.en"):
    if os.path.exists('./whisper-medium-ft/model.safetensors'):
        print(f'Fine tuned model at {model_path} already exists')
        return
    
    print(f'Fine-tuned model not found. Creating model from LoRA configuration at {lora_config}')
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(model, lora_config)

    mdoel = model.merge_and_unload()
    model.save_pretrained(model_path)
    
    print(f'Whisper model from {lora_config} saved at {model_path}')
    return

# Where fsp.py expects to find our fine-tuned model
model_path = 'whisper-medium-ft'
lora_config = './lora_config'

# Uncheck when uploaded to hf
load_whisper_model(model_path=model_path, lora_config=lora_config)

###### Helper functions #######

def format_metadata_header(filename, metadata, explicit_word_count):
    title, artist, album, year = metadata.get('title', 'N/A'), metadata.get('artist', 'N/A'), metadata.get('album', 'N/A'), metadata.get('year', 'N/A')
    genius_url, wer_score = metadata.get('genius_url'), metadata.get('wer_score')
    genius_link = f"|| **[View lyrics on Genius]({genius_url})**" if genius_url else ""
    wer_display = f"| similarity score = {wer_score} (lower is better)" if wer_score and genius_url else ""
    
    status_message = ""
    if explicit_word_count == 0:
        status_message = "\n\n**✅ No explicit content found in this track.**"
        
    return f"### Details for: *{filename}*\n**Artist:** {artist} | **Song:** {title} | **Album:** {album} ({year}) {genius_link} {wer_display}{status_message}"

def generate_static_transcript(transcript_data, initial_times):
    initial_times_set = {f"{t['start']}-{t['end']}" for t in initial_times}
    table_header = "<table><thead><tr><th style='width: 125px;'>Time</th><th>Line transcript</th><th>Explicit flag(s)</th></tr></thead><tbody>"
    table_rows = []

    all_lines = [" ".join([word['text'] for word in segment.get('line_words', [])]) for segment in transcript_data]

    explicit_results = []
    if all_lines:
        pipeline_outputs = tox_pipe(all_lines)
        
        for line_result in pipeline_outputs:
            flags = []

            for d in line_result:
                label = d['label']
                score = d['score']

                if score < 0.5: continue
                elif label == 'confilctual' or label == 'selfharm': flags.append('violence')
                elif label == 'profanity': flags.append('curse')
                elif label == 'drugs': flags.append('drugs')
                elif label == 'sex': flags.append('sex')

            explicit_results.append(flags)  

    for i, segment in enumerate(transcript_data):
        start_time_str, end_time_str = seconds_to_minutes(segment.get('start')), seconds_to_minutes(segment.get('end'))
        
        explicit_flag = ""

        if explicit_results:
            for flags in explicit_results[i]:
                if 'violence' in flags: explicit_flag += '💥'
                if 'curse' in flags: explicit_flag += '🤬'
                if 'drugs' in flags: explicit_flag += '🚬'
                if 'sex' in flags: explicit_flag += '🔞'
        
        words_in_line = segment.get('line_words', [])
        formatted_words = []

        for word in words_in_line:
            word_id = f"{word['start']}-{word['end']}"

            if word_id in initial_times_set:
                formatted_words.append(f"<s>{html.escape(word['text'])}</s>")

            else:
                formatted_words.append(html.escape(word["text"]))

        formatted_line = " ".join(formatted_words)
        table_rows.append(f"<tr><td>{start_time_str} - {end_time_str}</td><td>{formatted_line}</td><td style='text-align:center'>{explicit_flag}</td></tr>")
        
    return table_header + "".join(table_rows) + "</tbody></table>"

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
        analysis_state = analyze_audio(audio_file.name, model, device, fine_tuned, progress=None)
        all_results[filename] = analysis_state

    file_list = list(all_results.keys())
    first_file_results = all_results[file_list[0]]
    explicit_count_first_file = len(first_file_results['initial_explicit_times'])
    header = format_metadata_header(file_list[0], first_file_results['metadata'], explicit_count_first_file)
    transcript_html = generate_static_transcript(first_file_results['transcript'], first_file_results['initial_explicit_times'])
    
    # Check if ANY file has explicit content to determine if the apply button should be active
    any_explicit_content = any(len(res['initial_explicit_times']) > 0 for res in all_results.values())
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

def update_details_view(selected_filename, all_results):
    if not selected_filename or not all_results:
        return "", ""
    
    file_results = all_results[selected_filename]
    explicit_word_count = len(file_results['initial_explicit_times'])
    header = format_metadata_header(selected_filename, file_results['metadata'], explicit_word_count)
    transcript_html = generate_static_transcript(file_results['transcript'], file_results['initial_explicit_times'])
    return header, transcript_html

def handle_batch_finalization(all_results, progress=gr.Progress()):
    if not all_results:
        raise gr.Error("No active analysis session. Please process files first.")

    output_paths = []
    num_files = len(all_results)
    for i, (filename, analysis_state) in enumerate(all_results.items()):
        progress((i + 1) / num_files, desc=f"Applying edits {i + 1} of {num_files}")
        times_to_censor = analysis_state.get('initial_explicit_times', [])
        output_path = apply_censoring(analysis_state, times_to_censor, progress=None)
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

# MODIFIED: This function now accepts the analysis state to perform cleanup.
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
        gr.update(visible=True),   # upload_view
        gr.update(visible=False),  # review_view
        gr.update(visible=False),  # final_view
        gr.update(visible=True, interactive=True),   # apply_button
        gr.update(choices=[], value=None, visible=True), # processed_files_selector
        None,                      # analysis_results_state
        "",                        # details_header
        "",                        # transcript_output
        "",                        # final_status_output
        None,                      # edited_files_output
        None                       # files_input (to clear it)
    )


######  Gradio UI Definition  ########
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
            gr.Markdown("### How to use:")
            gr.Markdown('- Upload one or more audio files using the box below. Most common audio formats are accepted (e.g., `.mp3`, `.wav`, etc.).')
            gr.Markdown(f'- Click the **Process audio** button to create the transcriptions of the uploaded track(s). You will have a chance to review the edits before applying the censoring.')
        
            files_input = gr.File(label="Upload audio files", file_count="multiple", elem_id="upload-view", file_types=["audio"])
            process_button = gr.Button("Process audio", elem_id="upload-view")
        
            gr.Markdown('---')
            gr.Markdown('### How it works:')
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
            gr.Markdown("**Note**: Whisper's processing is not deterministic and it can sometimes get confused annd hallucinate with audio. If your transcription seems inaccurate (e.g., a line contains the same word repeated *many* times, or a line contains a significant amount of transcribed text not present in the song), please try running the program again on that song.")
            
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

    # --- Event Handlers ---
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

    # MODIFIED: The click handler now passes the analysis state to the cleanup function.
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
        js="() => confirm('Are you sure you want to return to the start? All current analysis will be lost.')"
    )

demo.launch(share=True)