import subprocess
import time
import requests
import webbrowser

# The URL of your Flask frontend
APP_URL = "http://127.0.0.1:7860/"

def start_app():
    """
    Starts the Docker containers, waits for the web app to be ready,
    and then opens it in a web browser.
    """
    # 1. Start the Docker containers in the background (detached mode)
    print("Starting Docker containers with 'docker compose up -d'...")
    try:
        subprocess.run(["docker", "compose", "up", "-d", "--build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Docker containers: {e}")
        print("Please ensure Docker Desktop is running.")
        return
    except FileNotFoundError:
        print("Error: 'docker' command not found. Is Docker Desktop installed and in your PATH?")
        return

    # 2. Wait for the Flask app to become responsive
    print(f"Waiting for the application to be ready at {APP_URL}...")
    for i in range(60):  # Wait for a maximum of 60 seconds
        try:
            response = requests.get(APP_URL, timeout=2)
            if response.status_code == 200:
                print("Application is ready!")
                break
        except requests.ConnectionError:
            # Server is not up yet, wait and try again
            time.sleep(1)
    else:
        print("Application did not start in time. Check the logs for errors:")
        print("  docker compose logs -f")
        return

    # 3. Open the URL in the default web browser
    print(f"Opening {APP_URL} in your browser...")
    webbrowser.open(APP_URL)

if __name__ == "__main__":
    start_app()