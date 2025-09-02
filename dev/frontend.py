# frontend.py

import gradio as gr
import requests  # To communicate with the FastAPI backend
import os
import shutil
import html
from datetime import datetime

# --- Configuration ---
# The URL where your FastAPI backend is running.
# If running on the same machine, this is the default.
BACKEND_URL = "http://127.0.0.1:8000"

# A local directory to store the final downloaded files from the backend
DOWNLOADS_DIR = "final_downloads"

# --- UI Helper Functions

def seconds_to_minutes(time):
    mins = int(time // 60)
    secs = int(time % 60)

    if secs == 0: return f'{mins}:00'
    elif secs < 10: return f'{mins}:0{secs}'
    else: return f"{mins}:{secs}"

def format_metadata_header(filename, metadata, explicit_word_count):
    """Metadata display for the full transcriptions. Includes genius link if possible."""
    title, artist, album, year = metadata.get('title', 'N/A'), metadata.get('artist', 'N/A'), metadata.get('album', 'N/A'), metadata.get('year', 'N/A')
    genius_url, wer_score = metadata.get('genius_url'), metadata.get('wer_score')
    genius_link = f"|| **[View lyrics on Genius]({genius_url})**" if genius_url else ""
    wer_display = f"| similarity score = {wer_score} (lower is better)" if wer_score and genius_url else ""
    
    status_message = ""
    if explicit_word_count == 0:
        status_message = "\n\n**✅ No explicit content found in this track.**"
        
    return f"### Details for: *{filename}*\n**Artist:** {artist} | **Song:** {title} | **Album:** {album} ({year}) {genius_link} {wer_display}{status_message}"

def generate_static_transcript(transcript_data, initial_ids, line_errs):
    """Generates the HTML table for the transcript view."""
    ids_to_mute = {tuple(item) for item in initial_ids}
    
    table_rows = []
    
    # Define headers based on whether there are LLM errors
    if not line_errs:
        table_header = "<table><thead><tr><th style='width: 125px;'>Time</th><th>Line transcript</th></thead><tbody>"
    else:
        table_header = "<table><thead><tr><th style='width: 125px;'>Time</th><th>Line transcript</th><th>LLM error</th></thead><tbody>"
    
    # Build each row
    for i, segment in enumerate(transcript_data):
        start_time_str = seconds_to_minutes(segment.get('start', 0))
        end_time_str = seconds_to_minutes(segment.get('end', 0))
        
        words_in_line = segment.get('line_words', [])
        formatted_words = []

        for word in words_in_line:
            word_id = tuple(word.get('id')) # JSON converts tuples to lists, so convert back
            if word_id in ids_to_mute:
                formatted_words.append(f"<s>{html.escape(word['text'])}</s>")
            else:
                formatted_words.append(html.escape(word["text"]))
        
        formatted_line = " ".join(formatted_words)
        
        if not line_errs:
            table_rows.append(f"<tr><td>{start_time_str} - {end_time_str}</td><td>{formatted_line}</td></tr>")
        else:
            llm_marker = '⚠️' if i in line_errs else ''
            table_rows.append(f"<tr><td>{start_time_str} - {end_time_str}</td><td>{formatted_line}</td><td>{llm_marker}</td></tr>")
            
    return table_header + "".join(table_rows) + "</tbody></table>"


# --- Modified Event Handlers (Client-Side Logic) ---

def handle_batch_analysis(files, progress=gr.Progress()):
    """
    Sends files to the backend for analysis and updates the UI with the results.
    """
    if not files:
        raise gr.Error("Please upload one or more audio files.")

    yield gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), None, None, None, None, None

    all_results = {}
    num_files = len(files)

    for i, audio_file in enumerate(files):
        filename = os.path.basename(audio_file.name)
        progress((i + 1) / num_files, desc=f"Uploading & Analyzing: {filename}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending '{filename}' to backend for analysis...")
        
        try:
            # Prepare the file for the POST request and send it to the backend
            with open(audio_file.name, 'rb') as f:
                files_payload = {'file': (filename, f, 'audio/mpeg')}
                response = requests.post(f"{BACKEND_URL}/analyze/", files=files_payload, timeout=600) # 10 minute timeout
            
            # Handle potential errors from the backend
            if response.status_code != 200:
                error_detail = response.json().get('detail', response.text)
                raise gr.Error(f"Error analyzing '{filename}': {error_detail}")
            
            # Store the JSON response from the backend
            result_data = response.json()
            job_id = result_data['job_id']
            
            # We store the job_id to use it later for finalization
            all_results[filename] = {
                "job_id": job_id,
                **result_data  # Unpack the rest of the data from the backend
            }
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Analysis complete for '{filename}'. Job ID: {job_id}")

        except requests.exceptions.RequestException as e:
            raise gr.Error(f"Could not connect to the backend at {BACKEND_URL}. Is it running? Error: {e}")

    # Update UI with the first file's results
    file_list = list(all_results.keys())
    if not file_list:
        raise gr.Error("Analysis failed for all files.")
        
    first_file_results = all_results[file_list[0]]
    explicit_count_first_file = len(first_file_results['initial_explicit_ids'])
    header = format_metadata_header(file_list[0], first_file_results['metadata'], explicit_count_first_file)
    transcript_html = generate_static_transcript(first_file_results['transcript'], first_file_results['initial_explicit_ids'], first_file_results['line_errs'])
    
    # Enable the apply button only if there's content to censor
    any_explicit_content = any(len(res['initial_explicit_ids']) > 0 for res in all_results.values())
    apply_button_update = gr.update(interactive=True, value="Apply all edits") if any_explicit_content else gr.update(interactive=False, value="No edits to make")

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
    """Updates the transcript view when a different file is selected from the radio list."""
    if not selected_filename or not all_results:
        return "", ""
    
    file_results = all_results[selected_filename]
    explicit_word_count = len(file_results['initial_explicit_ids'])
    header = format_metadata_header(selected_filename, file_results['metadata'], explicit_word_count)
    transcript_html = generate_static_transcript(file_results['transcript'], file_results['initial_explicit_ids'], file_results['line_errs'])
    return header, transcript_html

def handle_batch_finalization(all_results, progress=gr.Progress()):
    """
    Sends job IDs to the backend for finalization and provides the edited files for download.
    """
    if not all_results:
        raise gr.Error("No active analysis session. Please process files first.")

    output_paths = []
    num_files = len(all_results)
    
    # Ensure the local download directory exists and is empty
    if os.path.exists(DOWNLOADS_DIR):
        shutil.rmtree(DOWNLOADS_DIR)
    os.makedirs(DOWNLOADS_DIR)

    for i, (filename, analysis_data) in enumerate(all_results.items()):
        progress((i + 1) / num_files, desc=f"Finalizing '{filename}'")
        job_id = analysis_data['job_id']
        
        # Skip files with no explicit content
        if not analysis_data['initial_explicit_ids']:
            print(f"Skipping '{filename}' (no explicit content).")
            continue
            
        try:
            # Make the API call to the /finalize/ endpoint
            response = requests.post(f"{BACKEND_URL}/finalize/", json={"job_id": job_id}, timeout=300)
            
            if response.status_code == 200:
                # Save the received audio file to our local downloads directory
                base_name = os.path.splitext(filename)[0]
                output_filename = os.path.join(DOWNLOADS_DIR, f"{base_name}-edited.mp3")
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                output_paths.append(output_filename)
            else:
                error_detail = response.json().get('detail', response.text)
                print(f"Error finalizing '{filename}': {error_detail}")

        except requests.exceptions.RequestException as e:
            print(f"Connection error finalizing '{filename}': {e}")

    status_message = f"✅ **Success!** {len(output_paths)} of {len(all_results)} files have been censored and are ready for download."

    yield (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        status_message,
        output_paths, # This is a list of local paths
        gr.update(visible=True),
        gr.update(visible=False)
    )

def return_to_start(all_results):
    """Cleans up local files and resets the UI to its initial state."""
    # Clean up the local downloads directory
    if os.path.exists(DOWNLOADS_DIR):
        try:
            shutil.rmtree(DOWNLOADS_DIR)
            print(f"Cleaned up downloads directory: {DOWNLOADS_DIR}")
        except Exception as e:
            print(f"Error removing downloads directory {DOWNLOADS_DIR}: {e}")
            
    # NOTE: The temporary files on the backend are cleaned up when the backend server is shut down.
                    
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True, interactive=True),
        gr.update(choices=[], value=None, visible=True),
        None, "", "", "", None, None
    )

# --- Gradio UI Layout (copied from original script) ---

css = """
#main-container { max-width: 1250px; margin: auto; }
#main-container .prose { font-size: 15px !important; }
#upload-view { max-width: 60%; margin: 0 auto; }
#loading-view { min-height: 500px; display: flex; justify-content: center; align-items: center; }
#apply-button { background-color: #3d9c3e !important; color: white !important; }
#processed-files-radio { min-height: 300px; }
s { color: #d32f2f; text-decoration: line-through; }
"""

js_confirm_reset = """
() => {
  if (confirm('Are you sure you want to return to the start? All current analysis will be lost.')) {
    // Find the hidden button by its elem_id and click it
    document.getElementById('hidden_return_button').click();
  }
}
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
            gr.Markdown("""
                        This app uses a fine-tuned version of OpenAI's automatic speech recognition model [Whisper](https://github.com/openai/whisper) to create lyrics transcripts of the uploaded music files. Explicit words and phrases are as follows. 
                        
                        - Always explicit content (e.g., profanity, slurs) is immediately censored. 
                        - A language model ([Google Gemma 2](https://huggingface.co/google/gemma-2-9b-it)) then detects edge cases (e.g., drug references, sexually inappropriate content).
                        
                        Explicit content is highlighted in red strikehtough text in the full transcript. The vocals stem of the track is split off from the song using [demucs](https://github.com/facebookresearch/demucs) and muted at the appropriate times. The result is a high quality edited track, ready for air play.
                        """)

        with gr.Column(visible=False) as review_view:
            gr.Markdown("### Review transcript(s) and apply edits")
            gr.Markdown(f'Words to be censored will appear in <caption><s style="color: #d32f2f;">{html.escape("red strikethrough")}</s></caption> text in the transcript below. Click **Apply all edits** to apply the edits, this will reveal a link to download your edited file(s).')
            gr.Markdown("""
                        **Important** Language models are not deterministic and can fail in various ways. 
                            
                        - If you find that a portion of the lyrics has not been transcribed correctly/at all, please consider running the tool again on that song.
                        - If you see a column titled "LLM Error", then any rows containing the symbol "⚠️" were unable to be processed by the language model for edge-case explicit content. "Always explicit" content will still be censored, but this row may be missing censoring tags on additional edge-case content.

                        Always check your edited files against an official source to make sure all of the desired explicit content has been detected and censored.    
                        """)
            
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    processed_files_selector = gr.Radio(label="Select a file to view its transcript", interactive=True, elem_id="processed-files-radio")
                    apply_button = gr.Button("Apply all edits", elem_id="apply-button", interactive=False)
                    # return_to_start_button = gr.Button("Return to start")
                    # hidden_return_button = gr.Button("hidden_return", visible=False, elem_id="hidden_return_button")
                    with gr.Column(visible=False) as final_view:
                        final_status_output = gr.Markdown()
                        edited_files_output = gr.File(label="Download your edited files", file_count="multiple")

                with gr.Column(scale=3):
                    details_header = gr.Markdown()
                    with gr.Accordion("Full audio transcript", open=True):
                        transcript_output = gr.HTML()

        with gr.Column(visible=False, elem_id="loading-view") as loading_view:
            gr.Markdown("## ⏳ Processing... please wait")

    # --- Event Listeners ---
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

#     return_to_start_button.click(
#         fn=return_to_start,
#         inputs=[analysis_results_state],
#         outputs=[
#             upload_view, review_view, final_view, apply_button, processed_files_selector,
#             analysis_results_state, details_header, transcript_output, final_status_output,
#             edited_files_output, files_input
#         ],
#         js="() => { if (confirm('Are you sure you want to return to the start? All current analysis will be lost.')) { return true; } else { return false; } }"
#     )
    
demo.launch(favicon_path='fav.png')