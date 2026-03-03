import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# session = ort.InferenceSession("./models/yolox_nano.onnx", providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
# INPUT_SIZE = 416


session = ort.InferenceSession("./models/yolox_m.onnx", providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
INPUT_SIZE = 640
CONF_THRES = 0.4
NMS_THRES = 0.45

grids = []
strides = []
for stride in [8, 16, 32]:
    grid_size = INPUT_SIZE // stride
    for row in range(grid_size):
        for col in range(grid_size):
            grids.append([col, row])
            strides.append(stride)
grids = np.array(grids, dtype=np.float32)
strides = np.array(strides, dtype=np.float32).reshape(-1, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)

    blob = cv2.dnn.blobFromImage(frame, 1.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    out = session.run(None, {input_name: blob})[0][0]

    out[:, 0:2] = (out[:, 0:2] + grids) * strides
    out[:, 2:4] = np.exp(out[:, 2:4]) * strides

    boxes = []
    scores = []
    class_ids = []

    for det in out:
        obj_conf = det[4]
        if obj_conf < CONF_THRES:
            continue

        class_id = np.argmax(det[5:])
        score = float(obj_conf * det[5 + class_id])
        if score < CONF_THRES:
            continue

        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        x = int((cx - bw / 2) / ratio)
        y = int((cy - bh / 2) / ratio)
        w_box = int(bw / ratio)
        h_box = int(bh / ratio)

        boxes.append([x, y, w_box, h_box])
        scores.append(score)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, NMS_THRES)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            name = COCO_CLASSES[class_ids[i]]
            label = f"{name} {scores[i]:.2f}"
            print(label)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOX", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()