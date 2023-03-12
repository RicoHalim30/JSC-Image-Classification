from flask import Flask, request, render_template, send_from_directory
import numpy as np
from PIL import Image
from keras.models import load_model
import os

# load model
model = load_model('model2.h5')
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # dapatkan file gambar dari form
    file = request.files['image']
    # save file ke temp location
    file_path = "tmp/" + file.filename
    file.save(file_path)
    # konversi gambar menjadi numpy array
    img = np.array(Image.open(file_path).resize((150, 150))) / 255.
    # tambahkan dimensi ke array
    img = np.expand_dims(img, axis=0)
    # lakukan prediksi menggunakan model
    pred = model.predict(img)
    # ambil index dengan probabilitas terbesar
    result = class_names[np.argmax(pred)]
    return render_template('result.html', result=result, image_file=file.filename)

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory('tmp', filename)

if __name__ == '__main__':
    # hapus semua file yang ada di folder tmp saat aplikasi dijalankan
    for filename in os.listdir('tmp'):
        file_path = os.path.join('tmp', filename)
        os.remove(file_path)
    app.run(debug=True)
