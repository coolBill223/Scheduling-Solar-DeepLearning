from flask import Flask, request, send_file, render_template, send_from_directory, render_template_string
import os
import subprocess
from flask import request
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="web/static", template_folder="web")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    # Ensure target directory exists
    save_dir = os.path.join("data", "raw_data")
    os.makedirs(save_dir, exist_ok=True)

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(save_dir, "uploaded_data.xlsx")
    file.save(filepath)

    print(f"[UPLOAD] File saved to: {filepath}")
    return "File uploaded successfully. Ready to train."


@app.route("/train")
def train_model():
    train_script = os.path.join(os.getcwd(), "exp", "train.py")
    result = subprocess.run(
        ["python", train_script],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode == 0:
        return "Training completed."
    else:
        return "Training failed.", 500


@app.route("/loss_curve")
def loss_curve_img():
    # Serve the loss curve image from checkpoints
    return send_file("checkpoints/loss_curve.png", mimetype="image/png")

@app.route("/pred_vs_actual")
def pred_img():
    # Serve the prediction vs actual plot from checkpoints
    return send_file("checkpoints/pred_vs_actual.png", mimetype="image/png")

@app.route("/train_result_page")
def train_result_page():
    try:
        with open("checkpoints/metrics.txt") as f:
            metrics = f.read()
    except FileNotFoundError:
        return "No training results found. Please run training first."

    html = f"""
    <h2>Training Results</h2>
    <p>{metrics.replace('\n', '<br>')}</p>
    <img src="/loss_curve" width="500"><br><br>
    <img src="/pred_vs_actual" width="500">

    """
    return render_template_string(html)

if __name__ == "__main__":
    app.run(debug=True)
