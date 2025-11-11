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
    "Normal Colon",
    "Colon Ulcerative Colitis",
    "Colon Polyps",
    "Colon Esophagitis"
]

# --- Validate model output size vs provided labels and normalize labels ---
try:
    model_output_size = int(model.output_shape[-1])
except Exception:
    model_output_size = None

if model_output_size is not None:
    if model_output_size != len(class_labels):
        logging.warning(
            f"Model output size ({model_output_size}) does not match number of class_labels ({len(class_labels)}).\n"
            "Adjusting `class_labels` to match model output size."
        )
        # Keep as many provided labels as possible, pad or truncate to match
        if model_output_size > len(class_labels):
            # extend with placeholder names
            class_labels = class_labels + [f"Class {i}" for i in range(len(class_labels), model_output_size)]
        else:
            # truncate
            class_labels = class_labels[:model_output_size]

# Ensure labels are unique (append suffix if necessary)
seen = {}
for i, lab in enumerate(class_labels):
    if lab in seen:
        # append a numeric suffix to disambiguate
        new_lab = f"{lab} ({seen[lab]+1})"
        seen[lab] += 1
        class_labels[i] = new_lab
    else:
        seen[lab] = 1

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

    # safety: ensure probs is a flat list of floats
    probs = [float(p) for p in probs]

    # If model output size known, pad/truncate probs to match labels
    if 'model_output_size' in globals() and model_output_size is not None:
        if len(probs) != model_output_size:
            logging.warning(f"Predict returned {len(probs)} scores but expected {model_output_size}; padding/truncating to match.")
            if len(probs) < model_output_size:
                probs = probs + [0.0] * (model_output_size - len(probs))
            else:
                probs = probs[:model_output_size]

    predicted_class_idx = int(np.argmax(probs))
    predicted_class = (
        class_labels[predicted_class_idx]
        if predicted_class_idx < len(class_labels)
        else f"Class {predicted_class_idx}"
    )
    # confidence as raw score between 0-1
    confidence = float(probs[predicted_class_idx])

    return predicted_class_idx, predicted_class, confidence, probs

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
        predicted_idx, predicted_class, confidence, probs = predict_image(file_path)

        # Hapus file sementara
        try:
            os.remove(file_path)
        except Exception:
            logging.warning(f"Could not remove temporary file: {file_path}")

        # Validate probs vs labels length before returning
        if len(probs) != len(class_labels):
            logging.warning(f"Length mismatch: probs({len(probs)}) vs labels({len(class_labels)}). Truncating/padding response to match labels.")
            if len(probs) < len(class_labels):
                probs = probs + [0.0] * (len(class_labels) - len(probs))
            else:
                probs = probs[: len(class_labels)]

        # Convert probs to 0..100 scale for frontend display
        probs_percent = [round(float(p) * 100, 2) for p in probs]

        # Log the result for easier debugging
        logging.info(f"Prediction idx={predicted_idx} class={predicted_class} confidence={confidence:.4f} probs={probs_percent}")

        # Kirim hasil ke frontend
        return jsonify({
            'predicted_index': int(predicted_idx),
            'predicted_class': predicted_class,
            'prediction': predicted_class,  # backward compatibility
            'confidence': round(confidence * 100, 2),
            'confidence_score': confidence,
            'probs': probs_percent,
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
