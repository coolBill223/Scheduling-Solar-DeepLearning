from flask import Flask, request, send_file, render_template, send_from_directory, render_template_string
import os
import subprocess

app = Flask(__name__, static_folder="web/static", template_folder="web")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    filepath = os.path.join("data", "raw_data", "uploaded_data.xlsx")
    file.save(filepath)
    return "File uploaded successfully. Ready to train."

@app.route("/train")
def train_model():
    try:
        train_script = os.path.join(os.getcwd(), "exp", "train.py")
        result = subprocess.run(
            ["python3", train_script],
            capture_output=True,
            text=True,
            check=True 
        )
        output = f"<pre style='color:green;'>Training completed.\n\nSTDOUT:\n{result.stdout}</pre>"
        return output
    except subprocess.CalledProcessError as e:
        output = f"<pre style='color:red;'>Training failed.\n\nSTDERR:\n{e.stderr or 'No stderr'}\n\nSTDOUT:\n{e.stdout or 'No stdout'}</pre>"
        return output, 500


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
