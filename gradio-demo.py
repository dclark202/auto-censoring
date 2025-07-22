import gradio as gr
import pandas as pd
import os
import torch
import whisper
from fsp import process_audio, default_curse_words 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'medium.en'
print(f"Loading Whisper model ({model_name}) onto {device}...")
model = whisper.load_model(model_name, device=device)
print("Model loaded successfully.")



# --- Helper Function ---
# This wrapper function interfaces between the Gradio UI and your backend script.
def process_audio_files_progressive(files, additional_curses_str):
    """
    Takes files from the Gradio input, processes them one by one, and yields
    updates to the UI after each file is complete.
    """
    if not files:
        yield None, pd.DataFrame(), "Please upload one or more audio files."
        return # Exit the generator

    num_files = len(files)
    # Prepare lists to accumulate results over the loop
    output_file_paths = []
    all_logs_list = []

    # Process the additional curses string into a set
    additional_curses = set(word.strip().lower() for word in additional_curses_str.split(',')) if additional_curses_str else set()

    for i, file in enumerate(files):
        original_filename = os.path.basename(file.name)
        status_message = f"⏳ ({i+1}/{num_files}) Processing: {original_filename}..."
        
        # Immediately update UI to show which file is being processed
        temp_log_df = pd.concat(all_logs_list, ignore_index=True) if all_logs_list else pd.DataFrame()
        yield output_file_paths, temp_log_df, status_message

        # --- MODIFICATION: Call process_audio directly with the file path ---
        # The new process_audio function handles one file and returns one dictionary.
        result = process_audio(audio_path=file.name, 
                               model=model,
                               device=device,
                               additional_curses=additional_curses, 
                               delete_splits=True)
        
        # If processing failed or was skipped for a file, continue to the next
        if not result:
            continue

        # --- Accumulate the results from this file ---
        if result.get('output_path'):
            output_file_paths.append(result['output_path'])
        
        if result.get('explicit_log'):
            log_df = pd.DataFrame(result['explicit_log'])
            log_df['source_file'] = original_filename # Add filename to each log entry
            all_logs_list.append(log_df)

    # --- After the loop, prepare the final outputs ---
    if not all_logs_list:
        final_log_df = pd.DataFrame({
            "Status": ["No explicit content was found in any of the uploaded file(s)."]
        })
    else:
        final_log_df = pd.concat(all_logs_list, ignore_index=True)
        # Reorder columns for better presentation
        final_log_df = final_log_df[['source_file', 'word', 'start', 'end', 'prob']]

    final_status = f"✅ Processing complete. {len(output_file_paths)} of {num_files} file(s) were edited."
    
    # Yield the final, complete results
    yield output_file_paths, final_log_df, final_status


# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title='FSP Finder') as demo:
    gr.Markdown(
        """
        # FSP Finder 

        ## AI powered explicit content detector

        Upload one or more audio tracks (`.mp3`, `.wav`, etc.). This tool will automatically detect and silence all explicit words found in the uploaded tracks.
        You will receive the edited audio files and a detailed log of all censored words.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input for audio files
            files_input = gr.File(
                label="Upload audio files",
                file_count="multiple",
                file_types=["audio"] # Allows all audio types supported by the backend
            )
            # Input for additional curse words
            additional_curses_input = gr.Textbox(
                label="Additional words to censor (separate by comma)",
                placeholder="e.g., heck, gosh, darn"
            )
            
            with gr.Accordion("Default curse words (click to view)", open=False):
                    # Format the set of words into a nice, comma-separated string
                    word_list_str = ", ".join(sorted(list(default_curse_words)))
                    gr.Markdown(word_list_str)

            process_button = gr.Button("Process audio tracks", variant="primary")
            status_output = gr.Textbox(label="status", interactive=False, placeholder="Waiting for files...")

        with gr.Column(scale=2):
            # Output for the edited files
            edited_files_output = gr.File(
                label="Edited audio files",
                file_count="multiple"
            )
            # Output for the explicit content log
            log_output = gr.DataFrame(
                label="Explicit content log",
                headers=["File", "Word", "Start Time", "End Time", "Confidence"],
                wrap=True
            )
    
    # Define the click action
    process_button.click(
        fn=process_audio_files_progressive,
        inputs=[files_input, additional_curses_input],
        outputs=[edited_files_output, log_output, status_output]
    )

demo.launch(share=True)