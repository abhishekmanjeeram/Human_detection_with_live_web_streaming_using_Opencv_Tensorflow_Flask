from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)

configfile = "ssd_mobilenet_v3_large_coco.pbtxt"
frozen_model = "frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(frozen_model, configfile)
Class_labels = []
with open("labels.txt", 'rt') as fpt:
    Class_labels = fpt.read().strip("\n").split("\n")

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

def generate_frames():
    while True:
        success, frame = cap.read()
        if success:
            ClassIndex, Confidence, bbox = model.detect(frame, confThreshold=0.55)
            if len(ClassIndex) != 0:
                for ClassInd, Confid, Boxes in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
                    if ClassInd == 1:  # Check if the detected object is a person
                        cv2.rectangle(frame, Boxes, (255, 0, 0), 2)
                        # Concatenate object name and confidence
                        label = f"Person: {round(Confid * 100, 2)}%"
                        cv2.putText(frame, label, (Boxes[0] + 10, Boxes[1] + 40), font, fontScale=font_scale,
                                    color=(0, 255, 0), thickness=3)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
