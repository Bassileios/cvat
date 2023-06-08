import json
import base64
from PIL import Image
import io
import ultralytics
from ultralytics import YOLO

def init_context(context):
    context.logger.info("Init context...  0%")
    ultralytics.checks()

    # Read the DL model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
    model = YOLO("yolov8x.pt")
    context.user_data.model = model

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    results = context.user_data.model(image)

    encoded_results = []
    for result in results:
    # result = results[0]
        result = result.cpu()
        boxes = result.boxes
        names = result.names
        for box in reversed(boxes):
            cls, conf = box.cls.squeeze(), box.conf.squeeze()
            if float(conf.tolist()) <= threshold:
                continue

            c = int(cls)
            label = names[c]
            encoded_results.append({
                'confidence': float(conf.tolist()),
                'label': label,
                'points': box.xyxy.squeeze().tolist(),
                'type': 'rectangle'
            })

    # print(encoded_results)

    return context.Response(body=json.dumps(encoded_results), headers={},
        content_type='application/json', status_code=200)
