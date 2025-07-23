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
print("-----------------")


def process_audio_files_progressive(files):
    """
    Takes files from the Gradio input, processes them one by one, and yields
    updates to the UI, including a live console log.
    """
    if not files:
        # Update yield to match the new number of outputs
        yield None, pd.DataFrame(), "Please upload one or more audio files.", ""
        return

    # --- Initial Setup ---
    num_files = len(files)
    output_file_paths = []
    all_logs_list = []
    console_log = "" 

    # --- Main Loop for Each File ---
    for i, file in enumerate(files):
        original_filename = os.path.basename(file.name)
        status_message = f"⏳ ({i+1}/{num_files}) Now processing: {original_filename}..."
        console_log += f"\n----- Processing File {i+1}: {original_filename} -----\n"
        # Yield status and the start of the log for this file

        yield output_file_paths, pd.concat(all_logs_list, ignore_index=True) if all_logs_list else pd.DataFrame(), status_message, console_log

        final_result_for_file = None
        
        for update in process_audio(audio_path=file.name, model=model, device=device, delete_splits=True):
            
            if isinstance(update, str):
                # If the update is a string, it's a log message.
                console_log += update + "\n"
                # Yield the updated console log to the UI in real-time.
                yield output_file_paths, pd.concat(all_logs_list, ignore_index=True) if all_logs_list else pd.DataFrame(), status_message, console_log
            
            elif isinstance(update, dict):
                # If it's a dictionary, it's the final result for this file.
                final_result_for_file = update
        
        # --- Accumulate results after the inner loop finishes ---
        if not final_result_for_file:
            continue
            
        if final_result_for_file.get('output_path'):
            output_file_paths.append(final_result_for_file['output_path'])
        
        if final_result_for_file.get('explicit_log'):
            log_df = pd.DataFrame(final_result_for_file['explicit_log'])
            log_df['source_file'] = original_filename
            all_logs_list.append(log_df)

        status_message = f"✅ ({i+1}/{num_files}) Finished: {original_filename}"
        current_log_df = pd.concat(all_logs_list, ignore_index=True) if all_logs_list else pd.DataFrame()
        yield output_file_paths, current_log_df, status_message, console_log

    # --- Final Output Preparation ---
    if not all_logs_list:
        final_log_df = pd.DataFrame({"Status": ["No explicit content found."]})

    else:
        final_log_df = pd.concat(all_logs_list, ignore_index=True)
        final_log_df = final_log_df[['source_file', 'word', 'start', 'end', 'prob']]

    final_status = f"✅ Processing complete. {len(output_file_paths)} of {num_files} file(s) were edited."
    
    # Yield the final, complete results to all components
    yield output_file_paths, final_log_df, final_status, console_log


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
        inputs=[files_input],
        outputs=[edited_files_output, log_output, status_output]
    )

demo.launch()