# app.py
from flask import Flask, render_template, request, Response, jsonify
import requests
import os, json
from urllib.parse import unquote
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, ReplyTo

app = Flask(__name__)

# The URL for your FastAPI backend
BACKEND_URL = "http://127.0.0.1:8000"


# Route for the homepage (upload form)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # Use .getlist() to handle multiple files with the same name 'file'
    files = request.files.getlist('file')
    
    if not files or files[0].filename == '':
        return "No selected files", 400

    all_results = []
    
    for file in files:
        if file:
            # Prepare the file payload to send to the backend
            files_payload = {'file': (file.filename, file.read(), file.content_type)}
            
            try:
                # Make the request to the FastAPI backend for each file
                response = requests.post(f"{BACKEND_URL}/analyze/", files=files_payload, timeout=600)
                response.raise_for_status()
                
                # Add the JSON result to our list of all results
                all_results.append(response.json())

            except requests.exceptions.RequestException as e:
                # Handle connection errors or bad responses
                return f"Error analyzing '{file.filename}': {e}", 500

    # Render the results page, passing the list of all results
    return render_template('results.html', results_list=all_results)

@app.route('/finalize', methods=['POST'])
def finalize():
    """
    Receives a job_id and a user-edited list of word IDs to censor.
    Sends this data to the backend to get the final edited audio file
    and serves it to the user as a download.
    """
    job_id = request.form.get('job_id')
    ids_json_str = request.form.get('ids_to_censor')

    if not job_id:
        return "Missing job_id", 400
    
    ids_to_censor = []
    if ids_json_str:
        try:
            ids_to_censor = json.loads(ids_json_str)
        except json.JSONDecodeError:
            # This handles cases where the string is not empty but is malformed.
            return "Invalid format for ids_to_censor.", 400

        
    try:
        # Create the payload for the backend, including the user's custom list
        payload = {
            "job_id": job_id,
            "ids_to_censor": ids_to_censor
        }
        response = requests.post(f"{BACKEND_URL}/finalize/", json=payload, timeout=300)
        response.raise_for_status()
        
        # --- Robust Filename Parsing ---
        filename = f"{job_id}-edited.mp3" # Default fallback filename
        content_disposition = response.headers.get('content-disposition')
        
        if content_disposition:
            if 'filename*=' in content_disposition:
                encoded_name = content_disposition.split("''")[-1]
                filename = unquote(encoded_name)
            elif 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
        
        # Serve the file content as a download
        return Response(
            response.content,
            mimetype='audio/mpeg',
            headers={'Content-Disposition': f'attachment;filename="{filename}"'}
        )
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR communicating with backend: {e}")
        return f"An error occurred while finalizing the file: {e}", 500

        

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Basic validation
        if not all([name, email, message]):
            return jsonify({'status': 'error', 'message': 'All fields are required.'}), 400

        try:
            # Create the Mail object without the reply_to argument
            email_message = Mail(
                from_email='contact@fspfinder.com',
                to_emails='contact@fspfinder.com',
                subject=f'FSP Finder - {name}',
                html_content=f'<strong>Name:</strong> {name}<br>'
                             f'<strong>Email:</strong> {email}<br><br>'
                             f'<strong>Message:</strong><p>{message}</p>')

            # Create a ReplyTo object and assign it to the message
            email_message.reply_to = ReplyTo(email, name)

            # Get the API key and send the message as before
            sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
            response = sg.send(email_message)
            
            return jsonify({'status': 'success', 'message': 'Thank you for your message!'})
        except Exception as e:
            print(e)
            return jsonify({'status': 'error', 'message': 'An error occurred while sending your message.'}), 500

    # For a GET request, just show the contact page
    return render_template('contact.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')