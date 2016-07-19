import re
import json
import os
import sys
import redis
import logging
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
from PIL import Image, ImageEnhance
from keras.models import model_from_json
from keras.optimizers import RMSprop

app = Flask(__name__)
cors = CORS(app)

# Get port from environment variable or choose 9099 as local default
port = int(os.getenv("PORT", 9099))

# Get Redis credentials
if "VCAP_SERVICES" in os.environ:
    services = json.loads(os.getenv("VCAP_SERVICES"))
    redis_env = services["p-redis"][0]["credentials"]
else:
    redis_env = dict(host="localhost", port=6379, password="")

# Connect to redis
try:
    r = redis.StrictRedis(**redis_env)
    r.info()
except redis.ConnectionError as e:
    print(e)
    r = None

def get_model(redis):
    """Get the most recent model from redis and compile it."""
    recent_model = sorted([m.group(0) for l in r.keys()
                 for m in [re.compile(".*_model").search(l.decode("UTF-8"))]
                 if m], reverse = True)[0]
    recent_weights = sorted([m.group(0) for l in r.keys()
                 for m in [re.compile(".*_weights").search(l.decode("UTF-8"))]
                 if m], reverse = True)[0]

    model = model_from_json(redis.get(recent_model).decode("UTF-8"))
    weights = redis.get(recent_weights)
    with open("mnist_mlp_weights.h5", "wb") as f:
        f.write(weights)
    model.load_weights("mnist_mlp_weights.h5")
    os.remove("mnist_mlp_weights.h5")
    model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop())
    return model

def convert_image(image):
    """Resize the image and then transform it to 1-d vector."""
    img = Image.open(image).convert("RGBA")
    img_array = np.asarray(img, dtype="int32")[:,:,3]
    img_bw = Image.fromarray(img_array).convert("L")
    resized_image = img_bw.resize((28,28), Image.ANTIALIAS)
    brightness = ImageEnhance.Brightness(resized_image)
    rescaled_data = np.asarray(brightness.enhance(2), dtype="int32") / 255
    stacked_data = np.vstack([rescaled_data.reshape(-1)])
    return stacked_data

@app.route("/")
def main():
    return "Hello World!"

@app.route("/prediction", methods=["POST"])
def prediction():
    """
    curl -i -X POST -F files=@four.png http://0.0.0.0:9099/prediction
    """
    if request.method == "POST":
        image = request.files["files"]
        data = convert_image(image)
        model = get_model(r)
        prediction = model.predict_classes(data)
        return str(prediction[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
