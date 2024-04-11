from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from starlette.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI()

MODEL_DIR = './runs/detect/train/weights/best.pt'

# Load the YOLO model
model = YOLO(MODEL_DIR)

@app.post("/detect")
async def detect_defects(image: UploadFile = File(...)):
    contents = await image.read()
    image_data = Image.open(BytesIO(contents))

    # Perform inference
    predict = model.predict(image_data)

    # Plot bounding boxes
    plotted = predict[0].render()

    if len(predict[0].labels) == 0:
        return {"message": "No defects detected"}

    image_bytes = BytesIO()
    plotted.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
