import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

model_road = YOLO("best (1).pt")
model_road.to("cuda")
model_det = YOLO('yolo11n.pt')
model_det.to('cuda')

first_frame_processed = False
mask_coordinates = None
cap = cv2.VideoCapture('cvtest.avi')

text = 'Объект: не обнаружен'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    if not first_frame_processed:
        results = model_road.predict(source=frame, conf=0.5)
        result = results[0]
        if result.masks is not None:
            mask_coordinates = result.masks.xy
            first_frame_processed = True

    if mask_coordinates is not None:
        for mask in mask_coordinates:
            mask = mask.astype(int)

            x, y, w, h = cv2.boundingRect(mask)

            cropped_frame = frame[y:y + h, x:x + w]

            results_inside_box = model_det.predict(source=cropped_frame, conf=0.5)

            if results_inside_box[0].boxes is not None and len(results_inside_box[0].boxes) > 0:
                text = 'Объект: обнаружен'
            else:
                text = 'Объект: не обнаружен'

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = "arial.ttf"
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)


    draw.text((10, 10), text, font=font, fill=(0, 0, 0))


    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()