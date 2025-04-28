from flask import Flask, request, send_file, render_template, send_from_directory, render_template_string, redirect,jsonify
import subprocess
import importlib
from werkzeug.utils import secure_filename
import json
import os
import sys
internal_dir = os.path.join(os.path.dirname(sys.executable), "_internal")
if internal_dir not in sys.path:
    sys.path.insert(0, internal_dir)
    
BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
USER_DIR = os.path.expanduser('~')
CHECKPOINT_DIR = os.path.join(USER_DIR, ".my_software_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "web", "static"),
    template_folder=os.path.join(BASE_DIR, "web")
)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form.html")
def form_page():
    import json
    try:
        with open(os.path.join(CHECKPOINT_DIR, "category_mappings.json"), encoding="utf-8") as f:
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
        upload_dir = os.path.join(BASE_DIR, "data", "raw_data")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, "Data.xlsx")
        file.save(file_path)
        return "Upload complete", 200
    return "Upload failed", 400



@app.route("/train")
def train_model():
    try:
        from exp.trainNew import train_all_models
        train_all_models()
        return "Training completed.", 200
    except Exception as e:
        return f"<h3>Training error:</h3><pre>{str(e)}</pre>", 500


@app.route("/loss_curve")
def loss_curve_img():
    # Serve the loss curve image from checkpoints
    img_path = os.path.abspath(CHECKPOINT_DIR,"loss_curve.png")
    return send_file(img_path, mimetype="image/png")

@app.route("/pred_vs_actual")
def pred_img():
    # Serve the prediction vs actual plot from checkpoints
    img_path = os.path.join(CHECKPOINT_DIR, "pred_vs_actual.png")
    if not os.path.exists(img_path):
        return "Prediction image not found. Please train the model first.", 404
    return send_file(img_path, mimetype="image/png")


@app.route("/pred_vs_actual_tree")
def pred_img_tree():
    img_path = os.path.join(CHECKPOINT_DIR, "pred_vs_actual_tree.png")
    return send_file(img_path, mimetype="image/png")

@app.route("/pred_vs_actual_lr")
def pred_img_lr():
    img_path = os.path.join(CHECKPOINT_DIR, "pred_vs_actual_lr.png")
    return send_file(img_path, mimetype="image/png")

@app.route("/pred_vs_actual_ga")
def pred_img_ga():
    img_path = os.path.join(CHECKPOINT_DIR, "pred_vs_actual_ga.png")
    if not os.path.exists(img_path):
        return "GA prediction image not found. Please train the GA model first.", 404
    return send_file(img_path, mimetype="image/png")


@app.route("/train_result_page")
def train_result_page():
    def render_metrics(name, pretty_name):
        metrics_path = os.path.join(CHECKPOINT_DIR, f"metrics_{name}.txt") if name != "mlp" else os.path.join(CHECKPOINT_DIR, "metrics_mlp.txt")
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
    {render_metrics("ga", "Genetic Algorithm")}

    """
    return render_template_string(html)

@app.route("/predict", methods=["POST"])
def predict():
    mapping_path = os.path.join(CHECKPOINT_DIR, "category_mappings.json")
    if not os.path.exists(mapping_path):
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

        print("[DEBUG] Received model:", model_type)
        print("[DEBUG] Received features:", features)
        
        prediction = predict_with_model(model_type, features)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
