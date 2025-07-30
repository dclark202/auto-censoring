import gradio as gr
import os
import torch
import whisper
import whisper_timestamped as whisper_t
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import time
import re
import html
import json
from fsp import analyze_audio, apply_censoring, default_curse_words, seconds_to_minutes

# --- Load Models and Set Up ---
model_path = 'whisper-medium-ft'
print('Loading toxicity classifier...')
tox_ft_path = "roberta_search"
tox_tokenizer = AutoTokenizer.from_pretrained(tox_ft_path)
tox_model = AutoModelForSequenceClassification.from_pretrained(tox_ft_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tox_model.to(device)
tox_pipe = pipeline("text-classification", model=tox_model, tokenizer=tox_tokenizer, device=device, top_k=2)

def is_explicit(s):
    if not s.strip(): return False
    try:
        result = tox_pipe(s)[0]
        for d in result:
            if d['label'] == 'LABEL_1': return d['score'] > 0.5
    except Exception: return False
    return False

def format_metadata_header(filename, metadata):
    title, artist, album, year = metadata.get('title', 'N/A'), metadata.get('artist', 'N/A'), metadata.get('album', 'N/A'), metadata.get('year', 'N/A')
    genius_url, wer_score = metadata.get('genius_url'), metadata.get('wer_score')
    genius_link = f"**[View lyrics on Genius]({genius_url})**" if genius_url else ""
    wer_display = f"| similarity score = {wer_score} (lower is better)" if wer_score else ""
    return f"### Details for: *{filename}*\n**Artist:** {artist} | **Song:** {title} | **Album:** {album} ({year})\n{genius_link} {wer_display}"

def generate_static_transcript(transcript_data, initial_times):
    initial_times_set = {f"{t['start']}-{t['end']}" for t in initial_times}
    table_header = "<table><thead><tr><th>Time</th><th>Line</th><th>Explicit Line</th></tr></thead><tbody>"
    table_rows = []
    for segment in transcript_data:
        start_time_str, end_time_str = seconds_to_minutes(segment.get('start')), seconds_to_minutes(segment.get('end'))
        words_in_line = segment.get('line_words', [])
        full_line_text = " ".join([word['text'] for word in words_in_line])
        explicit_flag = "⚠️" if is_explicit(full_line_text) else ""
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

def handle_batch_analysis(files, selected_model_name, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload one or more audio files.")

    yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), None, None, None, None

    try:
        model, fine_tuned = (whisper.load_model(selected_model_name, device=device), False) if selected_model_name != 'fine-tuned' else (whisper_t.load_model(model_path, device=device), True)
    except Exception as e:
        raise gr.Error(f"Error loading Whisper model: {e}")

    all_results = {}
    num_files = len(files)
    for i, audio_file in enumerate(files):
        progress((i + 1) / num_files, desc=f"Analyzing File {i + 1} of {num_files}")
        filename = os.path.basename(audio_file.name)
        analysis_state = analyze_audio(audio_file.name, model, device, fine_tuned, progress=None)
        all_results[filename] = analysis_state

    file_list = list(all_results.keys())
    first_file_results = all_results[file_list[0]]
    header = format_metadata_header(file_list[0], first_file_results['metadata'])
    transcript_html = generate_static_transcript(first_file_results['transcript'], first_file_results['initial_explicit_times'])

    yield gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), all_results, gr.update(choices=file_list, value=file_list[0]), header, transcript_html

def update_details_view(selected_filename, all_results):
    if not selected_filename or not all_results:
        return "", ""
    
    file_results = all_results[selected_filename]
    header = format_metadata_header(selected_filename, file_results['metadata'])
    transcript_html = generate_static_transcript(file_results['transcript'], file_results['initial_explicit_times'])
    return header, transcript_html

def handle_batch_finalization(all_results, progress=gr.Progress()):
    if not all_results:
        raise gr.Error("No active analysis session. Please process files first.")

    yield gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), None, None, None, None

    output_paths = []
    num_files = len(all_results)
    for i, (filename, analysis_state) in enumerate(all_results.items()):
        progress((i + 1) / num_files, desc=f"Applying Censorship to File {i + 1} of {num_files}")
        times_to_censor = analysis_state.get('initial_explicit_times', [])
        output_path = apply_censoring(analysis_state, times_to_censor, progress=None)
        if output_path:
            output_paths.append(output_path)
            
    status_message = f"✅ **Success!** {len(output_paths)} of {len(all_results)} files have been censored."

    # This final yield statement is updated
    yield (
        gr.update(visible=True),      # review_view
        gr.update(visible=False),     # loading_view
        gr.update(visible=True),      # final_view
        status_message,               # final_status_output
        output_paths,                 # edited_files_output
        gr.update(visible=True),      # processed_files_selector (REVERTED: stays visible)
        gr.update(visible=False)      # apply_button (still disappears)
    )

# --- Gradio UI Definition ---
css = """
#main-container { max-width: 1250px; margin: auto; }
#loading-view { min-height: 500px; display: flex; justify-content: center; align-items: center; }
#apply-button { background-color: #3d9c3e !important; color: white !important; }
s { color: #d32f2f; text-decoration: line-through; }
"""

with gr.Blocks(theme=gr.themes.Soft(), title="FSP Finder", css=css) as demo:
    analysis_results_state = gr.State(None)

    with gr.Column(elem_id="main-container"):
        gr.Markdown("# FSP Finder - AI Powered Explicit Content Detector")
        gr.Markdown("Detects and automatically censors explicit content in audio tracks. For source code and more details, visit our [github page](https://github.com/dclark202/auto-censoring).")
        gr.Markdown("---")

        with gr.Column(visible=True) as upload_view:
            gr.Markdown("### How to use:")
            gr.Markdown('- Upload one or more audio files in the box below. Most audio formats accepted (e.g., `.mp3`, `.wav`, etc.).')
            gr.Markdown("- Select a Whisper model (Our fine-tuned model `fine-tuned` is recommended).")
            gr.Markdown('- Click "Process audio" to transcribe the uploaded tracks. Explicit content will be highlighted in red in the transcript of each track. Please allow ~30--60s for each track to be processed.')
            gr.Markdown("- Note: Whisper's decoding is not deterministic, and it can sometimes get confused with audio. If a transcribed song appears to be inaccurate (e.g., a line may contain the same word repeated *many* times), please try running the program again on that song.")

            files_input = gr.File(label="Upload audio files", file_count="multiple", file_types=["audio"])
            whisper_model_selector = gr.Dropdown(label="Select Whisper model", choices=['medium.en', 'large-v3', 'fine-tuned'], value='fine-tuned', interactive=True)
            process_button = gr.Button("Process audio")
        
        with gr.Column(visible=False) as review_view:
            gr.Markdown("### Review transcripts and apply edits")
            gr.Markdown('- Edits are not applied until clicking "Apply all edits" below.')
            gr.Markdown('- Note: The "Explicit line" column will have a flag if the line was determined to be explicit using our fine-tuned [toxicity filter](https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel). This flag may be raised due to explicit content beyond curse words. We are currently working on allowing users to select additional words to censor from the full transcript.')
            with gr.Row(variant="panel"):
                # Left column for file list, button, and final downloads
                with gr.Column(scale=1):
                    processed_files_selector = gr.Radio(label="Select a file to view its transcript", interactive=True)
                    apply_button = gr.Button("Apply all edits", elem_id="apply-button")
                    with gr.Column(visible=False) as final_view:
                        final_status_output = gr.Markdown()
                        edited_files_output = gr.File(label="Download your edited files", file_count="multiple")

                # Right column for details
                with gr.Column(scale=3):
                    details_header = gr.Markdown()
                    with gr.Accordion("Full audio transcript", open=True):
                        transcript_output = gr.HTML()

        with gr.Column(visible=False, elem_id="loading-view") as loading_view:
            gr.Markdown("## ⏳ Processing... please wait")

    # --- Event Handlers ---
    process_button.click(
        fn=handle_batch_analysis,
        inputs=[files_input, whisper_model_selector],
        outputs=[upload_view, review_view, loading_view, analysis_results_state, processed_files_selector, details_header, transcript_output]
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

demo.launch(share=True)