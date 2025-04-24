# Fall Detector using YOLOv8

This project implements a fall detection system using Python and a YOLOv8-based object detection model. It analyzes live or recorded video to detect human falls, which can be useful for elderly care, workplace safety, and surveillance systems.

## Demo

![Fall Detection Demo](media/output.gif)

## Features

- Real-time fall detection using YOLOv8
- Simple interface for testing with webcam or video files
- Easy to modify and integrate with alert systems

## Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/erenertemden/fall-detector-py.git
   cd fall-detector-py
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download or ensure the `yolov8n.pt` model is in the root directory**

   If not included, download it from the [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics/releases) or train your own.

## Usage

### Webcam

```bash
python fall-detection.py
```

### Video File

Modify `fall-detection.py` to set the path to your video:

```python
cap = cv2.VideoCapture('path_to_your_video.mp4')
```

## File Structure

- `fall-detection.py` – Main script for running the fall detection logic.
- `yolov8n.pt` – Pre-trained YOLOv8 model for object detection.
- `requirements.txt` – Python dependencies.

## Example Output

The system draws bounding boxes around detected people and prints alerts when a fall is suspected.

## Contributing

Feel free to fork the repo, improve the detection logic (e.g., via pose estimation or motion analysis), and submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).