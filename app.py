from flask import Flask, request, jsonify
from flask.helpers import send_file
import numpy as np
import onnxruntime
import cv2
import json

app = Flask(__name__,
            static_url_path='/', 
            static_folder='web')

# Lade ONNX-Modell
ort_session = onnxruntime.InferenceSession("efficientnet_lite0_Opset17.onnx")
# Ermittle Input- und Output-Namen aus dem Modell
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Lade Labels (achte drauf, dass labels_map.txt richtig formatiert ist als JSON)
labels = json.load(open("labels_map.txt", "r"))

def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # Pixelwerte von [0,255] auf ungefähr [-1,1] normalisieren
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = left + out_width
    top = int((height - out_height) / 2)
    bottom = top + out_height
    img = img[top:bottom, left:right]
    return img

@app.route("/")
def indexPage():
    return send_file("web/index.html")    

@app.route("/analyze", methods=["POST"])
def analyze():
    content = request.files.get('0', '').read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pre_process_edgetpu(img, (224, 224, 3))
    img_batch = np.expand_dims(img, axis=0)       # NHWC -> (1, 224, 224, 3)
    img_batch = np.transpose(img_batch, (0, 3, 1, 2))  # Channels first (1, 3, 224, 224)
    
    results = ort_session.run([output_name], {input_name: img_batch})[0]
    top5_indices = results[0].argsort()[-5:][::-1]
    result_list = [{"class": labels[str(idx)], "value": float(results[0][idx])} for idx in top5_indices]

    return jsonify(result_list)

if __name__ == "__main__":
    app.run(debug=True)
