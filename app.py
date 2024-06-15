
import os
from flask import * 
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

input_directory=''

# Ensure the 'uploads' directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('home_page.html')

@app.route('/display')
def display():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        global input_directory

        directory=os.path.join(app.config['UPLOAD_FOLDER'], filename)

        input_directory=''

        input_directory+=str(directory)
        
        return render_template("streaming.html")


@app.route('/video_processing')
def gen_frames(): 

    # define some parameters
    conf_threshold = 0.5
    max_cosine_distance = 0.4
    nn_budget = None
    points = [deque(maxlen=32) for _ in range(1000)] # list of deques to store the points
    counter_A = 0
    counter_B = 0
    counter_C = 0
    start_line_A = (0, 480)
    end_line_A = (480, 480)
    start_line_B = (525, 480)
    end_line_B = (745, 480)
    start_line_C = (895, 480)
    end_line_C = (1165, 480)
        
    model = YOLO("yolov8s.pt")

    # Initialize the deep sort tracker
    model_filename = "config/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # load the COCO class labels the YOLO model was trained on
    classes_path = "config/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # create a list of random colors to represent each class
    np.random.seed(42)  # to get the same colors
    colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)
    
    video_cap = cv2.VideoCapture(input_directory)

    while True:
        try:
            success, frame = video_cap.read()
            overlay = frame.copy()
    
            # draw the lines
            cv2.line(frame, start_line_A, end_line_A, (0, 255, 0), 12)
            cv2.line(frame, start_line_B, end_line_B, (255, 0, 0), 12)
            cv2.line(frame, start_line_C, end_line_C, (0, 0, 255), 12)
            
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            # if there is no frame, we have reached the end of the video
            if not success:
                print("End of the video file...")
                break

            ############################################################
            ### Detect the objects in the frame using the YOLO model ###
            ############################################################

            # run the YOLO model on the frame
            results = model(frame)

            # loop over the results
            for result in results:

                # initialize the list of bounding boxes, confidences, and class IDs
                bboxes = []
                confidences = []
                class_ids = []

                # loop over the detections
                for data in result.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = data
                    x = int(x1)
                    y = int(y1)
                    w = int(x2) - int(x1)
                    h = int(y2) - int(y1)
                    class_id = int(class_id)

                    # filter out weak predictions by ensuring the confidence is
                    # greater than the minimum confidence
                    if confidence > conf_threshold:
                        bboxes.append([x, y, w, h])
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
            # get the names of the detected objects
            names = [class_names[class_id] for class_id in class_ids]

            # get the features of the detected objects
            features = encoder(frame, bboxes)
            # convert the detections to deep sort format
            dets = []
            for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
                dets.append(Detection(bbox, conf, class_name, feature))

            # run the tracker on the detections
            tracker.predict()
            tracker.update(dets)


            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                # get the bounding box of the object, the name
                # of the object, and the track id

                bbox = track.to_tlbr()
                track_id = track.track_id
                class_name = track.get_class()
                # convert the bounding box to integers
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # get the color associated with the class name
                class_id = class_names.index(class_name)
                color = colors[class_id]
                B, G, R = int(color[0]), int(color[1]), int(color[2])

                # draw the bounding box of the object, the name
                # of the predicted object, and the track id
                text = str(track_id) + " - " + class_name
                text=class_name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 3)
                # cv2.rectangle(frame, (x1 - 1, y1 - 20),
                #             (x1 + len(text) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                ############################################################
                ### Count the number of vehicles passing the lines       ###
                ############################################################
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # append the center point of the current object to the points list
                points[track_id].append((center_x, center_y))

                # cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
                
                # loop over the set of tracked points and draw them
                for i in range(1, len(points[track_id])):
                    point1 = points[track_id][i - 1]
                    point2 = points[track_id][i]
                    # if the previous point or the current point is None, do nothing
                    if point1 is None or point2 is None:
                        continue
                    
                    # cv2.line(frame, (point1), (point2), (0, 255, 0), 2)
                    
                # get the last point from the points list and draw it
                last_point_x = points[track_id][0][0]
                last_point_y = points[track_id][0][1]
                # cv2.circle(frame, (int(last_point_x), int(last_point_y)), 4, (255, 0, 255), -1)    

                # if the y coordinate of the center point is below the line, and the x coordinate is 
                # between the start and end points of the line, and the the last point is above the line,
                # increment the total number of cars crossing the line and remove the center points from the list
                if center_y > start_line_A[1] and start_line_A[0] < center_x < end_line_A[0] and last_point_y < start_line_A[1]:
                    counter_A += 1
                    points[track_id].clear()
                elif center_y > start_line_B[1] and start_line_B[0] < center_x < end_line_B[0] and last_point_y < start_line_A[1]:
                    counter_B += 1
                    points[track_id].clear()
                elif center_y > start_line_C[1] and start_line_C[0] < center_x < end_line_C[0] and last_point_y < start_line_A[1]:
                    counter_C += 1
                    points[track_id].clear()
                    

            # draw the total number of vehicles passing the lines
            cv2.putText(frame, "A", (10, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, "B", (530, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, "C", (910, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"{counter_A}", (270, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"{counter_B}", (620, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"{counter_C}", (1040, 483), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if not success:
                break
            else:
                ret,buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        except:
            print('its running in error part')
            return redirect(url_for('gen_frames'))
        
    video_cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# @app.route('/stop')
# def stop():
#     return render_template('stop.html')

# @app.route('/close')
# def closing():
#     video_cap.release()    
#     return 'video processing completed'

if __name__ == '__main__':
    app.run(debug=True)
