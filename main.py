from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
import cv2
import torch
import super_gradients.training.models
from image.utils import read_image_file, create_image_response

app = FastAPI()

# Initialize the model globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = super_gradients.training.models.get("yolo_nas_s", pretrained_weights="coco").to(device)

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
        
        if img_response:
            # Save temporary image for visualization
            cv2.imwrite("media/temp_image.jpg", image)
            response = create_image_response(predictions)
            # Clean up temporary file
            import os
            os.remove("media/temp_image.jpg")
            return response
        
        return JSONResponse(content={"predictions": predictions})
    
    except Exception as e:
        # Clean up temporary file in case of error
        import os
        if os.path.exists("media/temp_image.jpg"):
            os.remove("media/temp_image.jpg")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "YOLO-NAS Object Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)