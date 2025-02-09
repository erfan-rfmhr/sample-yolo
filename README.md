# YOLO-NAS Object Detection API

This project implements a FastAPI-based REST API for object detection using YOLO-NAS model.

## Setup

You can run this project either using Docker or directly on your system.

### Docker Setup (Recommended)

1. Build the Docker image:
```bash
docker build -t yolo-nas-api .
```

2. Run the container:
```bash
# If you have a GPU:
docker run --gpus all -p 8000:8000 yolo-nas-api

# If you don't have a GPU:
docker run -p 8000:8000 yolo-nas-api
```

The API will be available at http://localhost:8000

### Local Setup

#### Prerequisites
- Python 3.10
- CUDA-capable GPU (optional, but recommended for better performance)
- Conda (recommended for environment management)

### Environment Setup

1. Create and activate a new conda environment:
```bash
conda create -n yolo-api python=3.10
conda activate yolo-api
```

2. Install PyTorch with CUDA support (if you have a GPU):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
For CPU-only installation:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

3. Install other dependencies:
```bash
pip install super-gradients
pip install fastapi python-multipart uvicorn
pip install opencv-python pillow
```

## Running the API

1. Start the API server:
```bash
uvicorn main:app --reload
```
The API will be available at http://localhost:8000

2. Access the API documentation:
- Open http://localhost:8000/docs in your web browser
- This provides an interactive interface to test the API

## API Endpoints

### GET /
- Health check endpoint
- Returns a welcome message

### POST /predict
- Accepts image file uploads
- Returns detected objects with their bounding boxes, classes, and confidence scores

Example response:
```json
{
  "predictions": [
    {
      "box": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400
      },
      "class": "person",
      "score": 0.95
    }
  ]
}
```

## Testing the API

Using curl:
```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/predict
```

Using Python requests:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Troubleshooting

1. If you encounter CUDA-related errors:
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA availability:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```

2. If you get import errors:
   - Verify all dependencies are installed
   - Make sure you're using the correct conda environment
