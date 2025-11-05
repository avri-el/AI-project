const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const previewContainer = document.getElementById("previewContainer");
const resultSection = document.getElementById("resultSection");
const classLabel = document.getElementById("classLabel");
const confidence = document.getElementById("confidence");

// 📸 Preview otomatis saat user memilih gambar
fileInput.addEventListener("change", function () {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      previewContainer.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  } else {
    previewContainer.classList.add("hidden");
  }
});

// 🧠 Kirim file ke backend
document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  resultSection.classList.remove("hidden");
  classLabel.textContent = "🧠 Analyzing image...";
  confidence.textContent = "";

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      classLabel.textContent = `❌ Error: ${data.error}`;
      confidence.textContent = "";
    } else {
      classLabel.textContent = `✅ Class: ${data.predicted_class}`;
      confidence.textContent = `🔹 Confidence: ${data.confidence}%`;
    }
  } catch (err) {
    classLabel.textContent = "❌ Server Error!";
    confidence.textContent = "";
  }
});
