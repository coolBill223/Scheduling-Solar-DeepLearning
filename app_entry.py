import threading
import webview
from main import app

# Function to run Flask in background
def run_flask():
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    # Run Flask server in a background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Open a native webview window pointing to the local Flask app
    webview.create_window("Solar Installation Time Predictor", "http://127.0.0.1:5000")
