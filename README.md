# AI image-colorization
> A Streamlit-based AI application that can **colorize grayscale images**, **enhance them using Super-Resolution (ESRGAN)**, and simulate **Daltonization** for color vision deficiency support.

## 🚀 Features

* 🎨 **Image Colorization** – Convert grayscale images into color using a TensorFlow/Keras model.
* 🔍 **Super-Resolution Enhancement** – Improve image quality with PyTorch ESRGAN (RRDBNet).
* 👓 **Daltonization Simulation** – Generate images as seen by individuals with color vision deficiency.
* 🖼️ **User-Friendly UI** – Streamlit interface with upload, preview, and download options.
* 📥 **One-Click Download** – Save processed images directly.

## 🛠️ Tech Stack

* **Frontend/UI**: Streamlit
* **Deep Learning**:

  * TensorFlow/Keras (image colorization)
  * PyTorch (ESRGAN-based super-resolution)
* **Computer Vision**: OpenCV, NumPy
* **Image Handling**: PIL, BytesIO

## 📂 Project Structure

```
AI-Image-Processor/
├─ app.py                         # Main Streamlit app
├─ RRDBNet_arch.py                # ESRGAN architecture (PyTorch)
├─ models/
│   ├─ color_image_checkpoint.keras  # Trained TensorFlow colorization model
│   └─ RRDB_ESRGAN_x4.pth            # Pretrained ESRGAN weights
├─ requirements.txt               # Dependencies
└─ README.md                      # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Create and activate a virtual environment:

   ```bash
   conda create -n ai-image python=3.10 -y
   conda activate ai-image
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place your trained models inside the `models/` directory.

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### Steps in the app:

1. Upload a grayscale or color image (`.jpg`, `.jpeg`, `.png`).
2. The app will:

   * Convert grayscale → Colorized image (TensorFlow model)
   * Enhance resolution (PyTorch ESRGAN)
   * Generate Daltonized simulation
3. View **Original | Processed | Daltonized** side by side.
4. Download the processed image.

---

## 🧠 Models

* **Colorization Model**: Trained TensorFlow `.keras` model for grayscale → color mapping.
* **Super-Resolution Model**: Pretrained ESRGAN (RRDBNet) for upscaling ×4.
* **Daltonization Matrix**: Fixed correction matrix simulating color vision deficiency.

---

## ❗ Troubleshooting

* **Model not loading** → Check file paths in `app.py` for `.keras` and `.pth` models.
* **CUDA OOM error** → Use smaller images or run on CPU.
* **Washed out colors** → Retrain colorization model with perceptual loss.

---

## 📌 Roadmap

* [ ] Support video colorization + enhancement
* [ ] Add multiple Daltonization modes (Protanopia, Deuteranopia, Tritanopia)
* [ ] Deploy as a web app (FastAPI + Streamlit Cloud)

## Licence
This project is licensed under the MIT License - see Licence file for details.

