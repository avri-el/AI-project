from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging

# --- Inisialisasi Flask ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Load model ---
MODEL_PATH = "model/colon_diseases.h5"
model = load_model(MODEL_PATH)

# --- Daftar label kelas (ubah sesuai model kamu) ---
class_labels = [
    "Colon Adenocarcinoma",
    "Colon Benign Tissue",
    "Colon Malignant",
    "Normal Colon Tissue"
]

# --- Fungsi bantu untuk prediksi gambar ---
def predict_image(img_path):
    # Ukuran gambar disesuaikan dengan model (256x256)
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # model.predict returns batch; take first sample
    preds = model.predict(x)
    probs = preds[0].tolist()
    predicted_class_idx = int(np.argmax(probs))
    predicted_class = (
        class_labels[predicted_class_idx]
        if predicted_class_idx < len(class_labels)
        else f"Class {predicted_class_idx}"
    )
    # confidence as raw score between 0-1
    confidence = float(probs[predicted_class_idx])

    return predicted_class, confidence, probs

# --- Route utama (tampilkan HTML UI) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Route prediksi (dipanggil dari form HTML) ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dikirim'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'})

    # Validate and save file securely
    ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'bmp'}
    filename = secure_filename(file.filename)
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
    else:
        ext = ''

    if ext not in ALLOWED_EXT:
        return jsonify({'error': 'Tipe file tidak didukung. Gunakan PNG/JPG/JPEG/BMP.'})

    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    logging.info(f"Saved uploaded file to {file_path}")

    try:
        # Prediksi
        predicted_class, confidence, probs = predict_image(file_path)

        # Hapus file sementara
        try:
            os.remove(file_path)
        except Exception:
            logging.warning(f"Could not remove temporary file: {file_path}")

        # Kirim hasil ke frontend
        # Return both a human-readable percent and raw score for compatibility
        return jsonify({
            'predicted_class': predicted_class,
            'prediction': predicted_class,  # backward compatibility
            'confidence': round(confidence * 100, 2),
            'confidence_score': confidence,
            'probs': [round(float(p) * 100, 2) for p in probs],
            'labels': class_labels,
        })

    except Exception as e:
        # Hapus file jika terjadi error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)})

# --- Jalankan server ---
if __name__ == '__main__':
    app.run(debug=True)
