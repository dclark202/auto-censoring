import gradio as gr
import pandas as pd
import os
import torch
import whisper
import whisper_timestamped as whisper_t
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import time
from fsp import process_audio, default_curse_words

### Toxicity classifier 
print('Loading toxicity classifier...')
tox_model_name = "cardiffnlp/twitter-roberta-large-sensitive-multilabel"
tox_ft_path = "roberta_search" # Fine tuned by Shuo

tox_tokenizer = AutoTokenizer.from_pretrained(tox_ft_path)
tox_model = AutoModelForSequenceClassification.from_pretrained(tox_ft_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
tox_model.to(device) 
pipe = pipeline("text-classification", model=tox_model, tokenizer=tox_tokenizer, device=device, top_k=2) # top_2 returns both labels

## Returns True if "toxic" label is > .5, False otherwise
def is_explicit(s):
    toxic = 0

    for d in pipe(s)[0]:
        if d['label'] != 'LABEL_1': 
            continue

        # Select the one for LABEL_1 = Toxic (LABEL_0 is non)
        toxic = d['score']

    return toxic > 0.5 # This particular classifier is almost alaways effectively 0 or 1...
#######################

# Helper functions
def seconds_to_minutes(time_in_seconds):
    """Converts seconds to a 'mm m ss s' format."""
    if time_in_seconds is None:
        return "0m 0s"
    
    minutes = int(time_in_seconds // 60)
    seconds = int(time_in_seconds % 60)
    
    return f"{minutes}m {seconds}s"

def generate_output_views(file_results):
    """
    Generates HTML strings for the explicit log and full transcript.
    """
    if not file_results:
        # Return empty strings if there are no results to display
        return "", ""

    # Explicit log view (list of censored words)
    log_data = file_results.get('explicit_log', [])
    log_html = ""
    
    if not log_data:
        log_html = "<p><i>No explicit content found.</i></p>"
    
    else:
        log_html = "<table><thead><tr><th>Word</th><th>Start</th><th>End</th><th>Probability</th></tr></thead><tbody>"
        
        for item in log_data:
            log_html += f"<tr><td>{item.get('word','')}</td><td>{item.get('start','')}</td><td>{item.get('end','')}</td><td>{item.get('prob','')}</td></tr>"
        
        log_html += "</tbody></table>"

    # Full transcript view
    transcript_data = file_results.get('full_transcript', [])
    transcript_html = ""
    
    if not transcript_data:
        transcript_html = "<p><i>Transcript not available.</i></p>"
    
    else:
        explicit_words_in_file = {item['word'].lower() for item in log_data}
        
        transcript_html = "<table><thead><tr><th>Time</th><th>Line</th><th>Explicit</th></tr></thead><tbody>"
        
        for segment in transcript_data:
            start_time = seconds_to_minutes(segment.get('start'))
            end_time = seconds_to_minutes(segment.get('end'))
            line = segment.get('line', '').strip()
            
            line_lower = line.lower()
            contains_explicit = is_explicit(line_lower)

            explicit_flag = "⚠️" if contains_explicit else ""
            formatted_line = f"<strong>{line}</strong>" if contains_explicit else line

            transcript_html += f"<tr><td>{start_time} - {end_time}</td><td>{formatted_line}</td><td style='text-align:center'>{explicit_flag}</td></tr>"
        
        transcript_html += "</tbody></table>"

    return log_html, transcript_html

def format_metadata_header(filename, metadata):
    """Creates a formatted Markdown string for the details header."""
    title = metadata.get('title', 'N/A')
    artist = metadata.get('artist', 'N/A')
    album = metadata.get('album', 'N/A')
    year = metadata.get('year', 'N/A')
    genius_url = metadata.get('genius_url')
    wer_score = metadata.get('wer_score')

    # For fomr
    genius_link = f"**[View lyrics on Genius]({genius_url})**" if genius_url else ""
    wer_display = f"| similarity score = {wer_score} (0.0 = perfect match, 1.0 = completely wrong)" if wer_score else ""

    # Displayed over each track's info
    header = f"""
    ### Details for: *{filename}*
    **Artist:** {artist} | **Song:** {title} | **Album:** {album} ({year})\n
    {genius_link} {wer_display}
    """

    return header

def update_display(evt: gr.SelectData, all_results: dict):
    """
    Called when a user selects a file. Updates the header, log, and transcript views.
    """
    selected_filename = os.path.basename(evt.value)
    base_filename = selected_filename.replace("-edited.mp3", "")
    
    key_found = None
    
    for key in all_results.keys():
        if base_filename in key:
            key_found = key
            break
            
    if key_found:
        file_results = all_results.get(key_found)
        log_html, transcript_html = generate_output_views(file_results)
        header_md = format_metadata_header(key_found, file_results.get('metadata', {}))

        return header_md, log_html, transcript_html
    
    else:
        # Return empty views if results for the selected file are not found
        return "File details not found.", "", ""


###################################################################


model_path = 'whisper-medium-ft'

def process_audio_files(files, selected_model_name, progress=gr.Progress(track_tqdm=True)):
    """
    Takes files from the Gradio input, processes them, and yields final results.
    Shows a loading screen with a progress bar during processing.
    """
    # Show loading screen, hide main content, hide details view
    yield gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), None, "", {}, "", "", ""

    if not files:
        yield gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), None, "Please upload one or more audio files.", {}, "", "", ""
        return
    
    # Set device and some 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_files = len(files)
    output_file_paths = []
    all_results = {}
    
    start_time = time.time()

    # Load the selected whisper model
    try:
        if selected_model_name != 'fine-tuned':
            model = whisper.load_model(selected_model_name, device=device)
            fine_tuned = False
        
        else:
            model = whisper_t.load_model(model_path, device=device)
            fine_tuned = True
    
    except Exception as e:
        error_message = f"Error loading Whisper model '{selected_model_name}': {e}."
        
        # Show main view on error
        yield gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), None, f"**Error:** {error_message}", {}, "", "", ""
        
        return

    ## Process bar isn't working yet...#########################################################################
    for i, file in enumerate(files):
        # --- Progress Bar Update ---
        eta_str = "Calculating..."
        
        if i > 0:
            elapsed = time.time() - start_time
            avg_time_per_track = elapsed / i
            eta_seconds = avg_time_per_track * (num_files - i)
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"

        # Handle filenames
        original_filename = os.path.basename(file.name)
        
        # I want this to appear in the Loading Screen
        progress((i) / num_files, desc=f"Processing {i+1}/{num_files}: {original_filename} (ETA: {eta_str})")
        #########################################################################################################

        final_result_for_file = None
        
        for update in process_audio(audio_path=file.name, model=model, device=device, delete_splits=True, fine_tuned=fine_tuned):
            if isinstance(update, dict):
                final_result_for_file = update

        if not final_result_for_file:
            continue
            
        all_results[original_filename] = {'explicit_log': final_result_for_file.get('explicit_log', []), 
                                          'full_transcript': final_result_for_file.get('full_transcript', []),
                                          'metadata': final_result_for_file.get('metadata', {})}
        
        if final_result_for_file.get('output_path'):
            output_file_paths.append(final_result_for_file['output_path'])

    final_status = f"✅ **Processing complete:** {len(output_file_paths)} of {num_files} file(s) were edited in {seconds_to_minutes(time.time() - start_time)}."
    
    initial_header, initial_log, initial_transcript = "", "", ""
    
    details_visible = False
    
    if all_results:
        details_visible = True
        first_filename = next(iter(all_results))
        file_results = all_results[first_filename]
        initial_header = format_metadata_header(first_filename, file_results.get('metadata', {}))
        initial_log, initial_transcript = generate_output_views(file_results)

    # Hide loading screen, show main content and details with results
    yield gr.update(visible=True), gr.update(visible=False), gr.update(visible=details_visible), output_file_paths, final_status, all_results, initial_header, initial_log, initial_transcript


# Gradio interface CSS
css = """
#loading-view {
    justify-content: center;
    align-items: center;
    height: 70vh;
}

#loading-view .center-text {
    text-align: center;
}

#process-button {
    background-color: #3d9c3e !important;
    color: white !important;
    transition: background-color 0.2s;
}

#process-button:hover {
    background-color: #284f29 !important;
}
"""


## 
with gr.Blocks(theme=gr.themes.Soft(), title='FSP Finder', css=css) as demo:
    results_state = gr.State({})

    gr.Markdown(
        """
        # FSP Finder - AI powered explicit content detector
        
        Upload audio tracks to automatically detect and silence explicit words. Edited audio files with appear in the list on the right. Click an output file to view the list of censored words along with a full transcript.
        
        For source code and more details, visit our [github page](https://github.com/dclark202/auto-censoring).
        """
    )

    # --- Loading Screen View (hidden by default) ---
    with gr.Column(visible=False, elem_id="loading-view") as loading_view:
        gr.Markdown("# ⏳ Processing... Please Wait", elem_classes="center-text")
        

    # --- Main Application View (visible by default) ---
    with gr.Column(visible=True, elem_id="main-view") as main_view:
        with gr.Row():
            with gr.Column(scale=1):
                files_input = gr.File(label="Upload audio files", file_count="multiple", file_types=["audio"])
                whisper_model_selector = gr.Dropdown(label="Select Whisper Model",
                                                     choices=['medium.en', 'large-v3', 'fine-tuned'],
                                                     value='fine-tuned',
                                                     interactive=True)
                
                process_button = gr.Button("Process audio tracks", elem_id="process-button")
                
                with gr.Accordion("Words to censor (click to expand)", open=False):
                        
                        word_list_str = ", ".join(sorted(list(default_curse_words)))
                        
                        gr.Markdown(word_list_str)

            with gr.Column(scale=3):
                final_status_output = gr.Markdown()
                edited_files_output = gr.File(label="Edited audio files (Click a file to see details)", file_count="multiple")
                
                # --- Details view, hidden until processing is done ---
                with gr.Column(visible=False) as details_view:
                    details_header = gr.Markdown()
                   
                    with gr.Accordion("Censored words", open=True):
                        log_output = gr.HTML()

                    with gr.Accordion("Full audio transcript", open=True):
                        transcript_output = gr.HTML()


    # --- Event Handlers ---
    process_button.click(fn=process_audio_files,
                         inputs=[files_input, whisper_model_selector],
                         outputs=[main_view, loading_view, details_view, edited_files_output, final_status_output, results_state, details_header, log_output, transcript_output]
                         )

    edited_files_output.select(fn=update_display,
                               inputs=[results_state],
                               outputs=[details_header, log_output, transcript_output]
                               )

demo.launch(share=True)