
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("trained_weights_custom_data/yolov4-custom-monkey_best.weights", "yolov4-custom-monkey.cfg")
classes = []
with open("coco_custom.names", "r") as f:

    classes = [line.strip() for line in f.readlines()]
    print(classes)
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Load image
image = cv2.imread("Monkey_100_images/38.jpg")
image = cv2.resize(image, None, fx=0.4, fy=0.4)
height, width, channels = image.shape

# Ditecting Objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)


# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
colors = np.random.uniform(0, 255, size=(len(boxes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 1)
        cv2.putText(image, label + " " + confidence, (x, y + 10), font, 1, color, 1)


cv2.imshow("output",image)
if ord("s"):
    cv2.imwrite("output_image_3.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()