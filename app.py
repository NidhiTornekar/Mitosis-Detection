import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import cv2
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best (1).pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict(image_path):
    # Preprocess
    img = Image.open(image_path)
    img = img.convert('RGB')
    img.save('uploads/temp.jpg')
    img.save('static/temp.jpg')
    results = model('uploads/temp.jpg')
    preds = results.xyxy[0].numpy()
    img = cv2.imread('uploads/temp.jpg')
    for pred in preds: #drawing bounding boxes
        x_min, y_min, x_max, y_max = pred[:4]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
    cv2.imwrite('uploads/predicted.jpg', img)
    cv2.imwrite('static/predicted.jpg', img)
    filename = os.path.basename(image_path)
    return 'predicted.jpg', filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_image, uploaded_filename = predict(file_path)
            return render_template('result.html', predicted_image_path=predicted_image, uploaded_filename=uploaded_filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
