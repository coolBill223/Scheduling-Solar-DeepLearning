import threading
import webbrowser
import os
from main import app

def run_flask():
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    if not os.environ.get("FLASK_STARTED"):
        os.environ["FLASK_STARTED"] = "1"
        webbrowser.open("http://127.0.0.1:5000")
    input("Press ENTER to stop the server...\n")