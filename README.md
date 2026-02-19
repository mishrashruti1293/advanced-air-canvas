
# advanced-air-canvas

**Real-Time Hand Gesture Drawing Application**

advanced-air-canvas is a computer vision application that enables users to draw on a virtual canvas using hand gestures. Built with OpenCV and MediaPipe, the system tracks hand landmarks in real time and allows gesture-based drawing, shape creation, erasing, and dynamic brush control.

---

## Features

* Real-time hand tracking using MediaPipe
* Freehand drawing with index finger
* Dynamic brush thickness control using pinch gesture
* Draw straight lines
* Draw rectangles
* Draw circles
* Erase using gesture control
* Save canvas as image
* Live FPS display for performance monitoring

---

##  Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy

---

## Project Structure

```
Advanced-Air-Canvas/
│
├── virtual_paint_app.py
├── tools.png (optional)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```
git clone <https://github.com/mishrashruti1293/advanced-air-canvas>
cd advanced-air-canvas
```

### 2. Create Virtual Environment (Recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```
pip install opencv-python mediapipe numpy
```

---

## ▶️ Running the Application

```
python virtual_paint_app.py
```

Make sure:

* Webcam is connected
* Camera permissions are enabled in system settings

---

## ✋ Gesture Controls

| Fingers Detected | Action        |
| ---------------- | ------------- |
| 1 Finger         | Freehand Draw |
| 2 Fingers        | Line          |
| 3 Fingers        | Rectangle     |
| 4 Fingers        | Circle        |
| 0 Fingers        | Erase         |

Pinch gesture dynamically adjusts brush thickness.

---

## ⌨️ Keyboard Controls

| Key | Function             |
| --- | -------------------- |
| S   | Save canvas as image |
| C   | Clear canvas         |
| ESC | Exit application     |

---

## 🔮 Future Enhancements

* Color palette selection
* Undo / Redo functionality
* Shape fill option
* Toolbar overlay interface
* Multi-hand support
* Stroke smoothing optimization

---

## 👩‍💻 Author

Shruti Mishra
B.Tech – Information Technology (Sp. Data Science)


If you find this project useful or interesting, consider starring the repository.



