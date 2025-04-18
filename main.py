from flask import Flask, request, send_file, render_template, send_from_directory, render_template_string, redirect,jsonify
import os
import subprocess
import importlib
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="web/static", template_folder="web")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form.html")
def form_page():
    import json
    try:
        with open("checkpoints/category_mappings.json", encoding="utf-8") as f:
            mappings = json.load(f)
    except:
        mappings = {}

    return render_template("form.html", mappings=mappings)



@app.route("/train.html")
def train_page():
    # New page to upload Excel, trigger training, and view results
    return render_template("train.html")


@app.route("/instructions.html")
def instructions_page():
    # Static instructions page
    return render_template("instructions.html")

@app.route("/index.html") 
def index_page():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get("file")
    if file and file.filename.endswith(".xlsx"):
        file_path = os.path.join("data", "raw_data", "Data.xlsx")
        file.save(file_path)
        subprocess.run(["python", "exp/trainNew.py"])

        
        return redirect("/train.html?status=success")
    return redirect("/train.html?status=fail")



@app.route("/train")
def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(BASE_DIR, "exp", "trainNew.py")
    result = subprocess.run(
        ["python", train_script],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode == 0:
        return "Training completed."
    else:
        return f"<h3>Training failed.</h3><pre>{result.stderr}</pre>", 500



@app.route("/loss_curve")
def loss_curve_img():
    # Serve the loss curve image from checkpoints
    return send_file("checkpoints/loss_curve.png", mimetype="image/png")

@app.route("/pred_vs_actual")
def pred_img():
    # Serve the prediction vs actual plot from checkpoints
    return send_file("checkpoints/pred_vs_actual.png", mimetype="image/png")

@app.route("/pred_vs_actual_tree")
def pred_img_tree():
    return send_file("checkpoints/pred_vs_actual_tree.png", mimetype="image/png")

@app.route("/pred_vs_actual_lr")
def pred_img_lr():
    return send_file("checkpoints/pred_vs_actual_lr.png", mimetype="image/png")

@app.route("/train_result_page")
def train_result_page():
    def render_metrics(name, pretty_name):
        metrics_path = f"checkpoints/metrics_{name}.txt" if name != "mlp" else "checkpoints/metrics.txt"
        image_path = f"/pred_vs_actual_{name}" if name != "mlp" else "/pred_vs_actual"

        try:
            with open(metrics_path) as f:
                metrics = f.read().replace('\n', '<br>')
        except FileNotFoundError:
            metrics = "No results found."

        return f"""
            <h3>{pretty_name}</h3>
            <p>{metrics}</p>
            <img src="{image_path}" width="500"><br><br>
        """

    html = f"""
    <h2>Training Results for All Models</h2>
    {render_metrics("mlp", "TinyMLP (Neural Networks)")}
    {render_metrics("tree", "Tree")}
    {render_metrics("lr", "Regression")}
    """
    return render_template_string(html)

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists("checkpoints/category_mappings.json"):
        return jsonify({"error": "Model not trained yet. Please train first."}), 400
    
    from models.predict_engine import predict_with_model
    
    try:
        data = request.get_json()
        model_type = data.get("model")

        if "feature_vector" in data:
            features = data["feature_vector"]
        else:
            selected_features = [
            "tilt", "azimuth", "panel_qty", "system_rating", "inverter_manufacturer", "array_type",
            "squirrel_screen", "consumption_monitoring", "truss_rafter", "reinforcements", "interconnection_type",
            "module_length", "module_width", "module_weight", "num_arrays", "num_circuits", "num_reinforcement",
            "roof_type", "attachment_type", "orientation", "num_stories", "install_season", "num_employees"
            ] 
            features = [float(data.get(k,0)) for k in selected_features]

        prediction = predict_with_model(model_type, features)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
