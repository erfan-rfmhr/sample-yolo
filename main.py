from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
import super_gradients.training.models
import io
from PIL import Image

app = FastAPI()

# Initialize the model globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = super_gradients.training.models.get("yolo_nas_s", pretrained_weights="coco").to(device)

def read_image_file(file_content) -> np.ndarray:
    # Convert the uploaded file to an image
    image = Image.open(io.BytesIO(file_content))
    # Convert PIL Image to cv2 image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...), img_response: bool = Query(default=False), num_of_classes: int = Query(default=None)):
    try:
        # Read the image file
        contents = await file.read()
        image = read_image_file(contents)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        results = model.predict(image)
        
        # Extract prediction details
        pred_boxes = results.prediction.bboxes_xyxy
        pred_classes = results.prediction.labels
        pred_scores = results.prediction.confidence
        class_names = results.class_names
        
        # Format results
        predictions = []
        
        if num_of_classes is not None:
            pred_boxes = pred_boxes[:num_of_classes]
            pred_classes = pred_classes[:num_of_classes]
            pred_scores = pred_scores[:num_of_classes]

        for box, class_id, score in zip(pred_boxes, pred_classes, pred_scores):
            predictions.append({
                "box": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3])
                },
                "class": class_names[int(class_id)],
                "score": float(score)
            })
        
        return JSONResponse(content={"predictions": predictions})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "YOLO-NAS Object Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)