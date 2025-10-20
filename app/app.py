from flask import Flask, render_template, request
import numpy as np
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from model import image_pre, predict

# ---- Initialize Flask app ----
app = Flask(__name__, static_folder='static', template_folder='templates')

# ---- Path configuration (cross-platform and portable) ----
# Use the Flask app root path as the base directory (points to app/)
BASE_DIR = Path(app.root_path)

# Define upload directory inside app/static
UPLOAD_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Automatically create folder if missing

# ---- App configuration ----
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)  # Convert Path to string for compatibility

# ---- Helper function ----
def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---- Main route ----
@app.route("/", methods=["GET", "POST"])
def index():
    """Render the main page and handle image upload."""
    result = ""
    if request.method == "POST":
        if "file1" not in request.files:
            return "No file part found in the form!"

        file1 = request.files["file1"]
        if file1.filename == "":
            return "No file selected!"

        if not allowed_file(file1.filename):
            return "File type not allowed!"

        # Save file as static/input.png (overwrite each time)
        save_path = UPLOAD_DIR / "input.png"
        file1.save(str(save_path))

        # Run preprocessing and prediction
        data = image_pre(str(save_path))
        s = predict(data)
        result = s if s else ""

    return render_template("index.html", result=result)

# ---- Run server ----
if __name__ == "__main__":
    app.run(debug=True)
