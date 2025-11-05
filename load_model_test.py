from tensorflow.keras.models import load_model
model = load_model("colon_diseases.h5")
model.summary()