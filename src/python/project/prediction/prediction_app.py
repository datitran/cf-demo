from flask import Flask, request
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

@app.route("/")
def main():
    return "Test"

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """
    curl -i -X POST -F files=@four_test.png http://127.0.0.1:5000/prediction
    """
    if request.method == "POST":
        f = request.files["files"]
        img = Image.open(f).convert("L")
        data = np.asarray(img, dtype="int32")
        return "200"
    else:
        return "201"

if __name__ == "__main__":
    app.run(debug=True)
