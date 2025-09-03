from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, cv2, numpy as np
from ultralytics import YOLO
from collections import Counter
import webbrowser
import threading

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLOv8 model (will download yolov8n.pt on first run)
model = YOLO("yolov8n.pt")


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return float(interArea) / union


def nms_per_class(detections, iou_thresh=0.5):
    kept = []
    groups = {}
    for d in detections:
        groups.setdefault(d['name'], []).append(d)

    for name, items in groups.items():
        items = sorted(items, key=lambda x: x['conf'], reverse=True)
        selected = []
        for it in items:
            skip = False
            for s in selected:
                if iou(it['box'], s['box']) > iou_thresh:
                    skip = True
                    break
            if not skip:
                selected.append(it)
        kept.extend(selected)

    kept = sorted(kept, key=lambda x: x['conf'], reverse=True)
    return kept


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        results = model(filepath, conf=0.45, iou=0.45)
        r = results[0]

        if len(r.boxes) == 0:
            return render_template("index.html", filename=filename, results={"total": 0, "counts": {}, "detected": []})

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        names_map = r.names

        detections = []
        for box, conf, cls in zip(boxes, confs, classes):
            name = names_map[int(cls)]
            detections.append({"name": name, "conf": float(conf), "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]})

        filtered = nms_per_class(detections, iou_thresh=0.5)

        detected_names = [d["name"] for d in filtered]
        counts = dict(Counter(detected_names))
        total = len(detected_names)

        img = cv2.imread(filepath)
        for d in filtered:
            x1, y1, x2, y2 = map(int, d["box"])
            label = f"{d['name']} {d['conf']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        annotated_name = "annot_" + filename
        annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], annotated_name)
        cv2.imwrite(annotated_path, img)

        results_out = {"total": total, "counts": counts, "detected": detected_names}
        return render_template("index.html", filename=annotated_name, results=results_out)

    return render_template("index.html")


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()  # open browser after 1 second
    app.run(debug=True)
