import threading
import webview
from main import app

# Function to run Flask in background
def run_flask():
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    # Start Flask server in background thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    print("Flask server started at http://127.0.0.1:5000")

    # Create a native window via pywebview
    webview.create_window(
        title="Solar Installation Time Predictor",
        url="http://127.0.0.1:5000",
        width=1280,
        height=800,
        resizable=True
    )

    # Start the GUI event loop
    webview.start()
