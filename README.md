<div align="center">
  <img src="assets/logo.png" alt="ISL Logo" width="150"/>

  # 🖐️ Indian Sign Language Detection (Web App)
  <p>
    <strong>Empowering communication for specially-abled children using AI 🤖</strong><br/>
    A real-time sign language recognition web app built with <b>TensorFlow.js</b> and <b>MediaPipe</b>.
  </p>

  <img src="https://img.shields.io/badge/Made%20With-TensorFlow.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-HTML%2FCSS%2FJS-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"/>
  <img src="https://img.shields.io/github/license/yourusername/isl-detection?style=for-the-badge"/>

  <br/>
  <a href="https://github.com/yourusername/isl-detection/stargazers">⭐ Star this repo</a> |
  <a href="https://github.com/yourusername/isl-detection/issues">🐛 Report Issues</a> |
  <a href="https://github.com/yourusername/isl-detection/fork">🍴 Fork</a>
</div>

---

## 📸 **Preview**

<table align="center">
<tr>
<td align="center"><img src="https://github.com/user-attachments/assets/30670772-ee4a-4653-8382-cac633f08530" width="350"/><br/><b>Real-time Hand Tracking</b></td>
<td align="center"><img src="assets/demo2.png" width="350"/><br/><b>Letter Prediction</b></td>
</tr>
</table>

---

## 🧠 **About the Project**

> This web application detects **Indian Sign Language (ISL)** hand gestures in real-time through your webcam.  
> It’s specially designed for **children who cannot speak**, offering a fun, educational way to learn communication using signs.  
> 
> Built with ❤️ using **TensorFlow.js**, **MediaPipe Hands**, and a custom-trained deep learning model on Indian Sign Language alphabets.

---

## ⚙️ **Features**

✅ Real-time hand tracking using Google MediaPipe  
✅ Predicts A–Z ISL letters instantly 🖐️  
✅ Smooth & interactive UI (kid-friendly)  
✅ Works directly in the browser — *no installation needed!*  
✅ Trained on a lightweight CNN model  
✅ Converts model from `.h5` → `.json` using TensorFlow.js  

---

## 🧩 **Tech Stack**

| Technology | Description |
|-------------|-------------|
| **TensorFlow.js** | For loading & running the trained deep learning model in the browser |
| **MediaPipe Hands** | For real-time hand tracking (21 landmarks) |
| **JavaScript (ES6)** | Handles frame capture & prediction logic |
| **HTML5 / CSS3** | Responsive front-end UI |
| **Python (TensorFlow/Keras)** | Model training and conversion to TFJS format |

---

## 🚀 **Getting Started**

### 🔹 1. Clone the Repository
```bash
git clone https://github.com/yourusername/isl-detection.git
cd isl-detection

