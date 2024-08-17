import cv2
import numpy as np
import time
from sort import *
from twilio.rest import Client
import keys

msg_count = 0
n_alert = False
s_alert = False
car_count_n = []
truck_count_n = []
bus_count_n = []
bike_count_n = []
van_count_n = []

car_count_s = []
truck_count_s = []
bus_count_s = []
bike_count_s = []
van_count_s = []

# Load Yolo
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("./detect.cfg", "./detect_final.weights")

# with GPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("./obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading camera
cap = cv2.VideoCapture("./video14.mp4")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    ret, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape[:3]

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()

    detections = []
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.50:
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
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    
    for i in range(len(boxes)):
        line_up = cv2.line(frame, (20,380),(350,380), (0,0,0),3)
        line_dn = cv2.line(frame, (450,600),(1250,600), (0,0,0),3)
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), font, 2, (255,0,0), 2)
            # # Add detection to the detections list
            detections.append([x, y, x + w, y + h, confidence])

    # Update tracker with detections
    detections = np.array(detections)
    results_tracker = tracker.update(detections)

    # Draw tracked objects
    for result in results_tracker:
        x1, y1, x2, y2, id = result.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame,(center_x,center_y),5,(255,0,255),cv2.FILLED)

        # For line_UP
        if 80<center_x<300 and 330<center_y<370 :
            if label == "car" and id not in car_count_n:
                car_count_n.append(id)
            elif label == "truck" and id not in truck_count_n:
                truck_count_n.append(id)
            elif label == "bike" and id not in bike_count_n:
                bike_count_n.append(id)
            elif label == "bus" and id not in bus_count_n:
                bus_count_n.append(id)
            elif label == "van" and id not in van_count_n:
                van_count_n.append(id)
        
        # For line_down
        if 500<center_x<1100 and 380<center_y<420 :
            if label == "car" and id not in car_count_s:
                car_count_s.append(id)
            elif label == "truck" and id not in truck_count_s:
                truck_count_s.append(id)
            elif label == "bike" and id not in bike_count_s:
                bike_count_s.append(id)
            elif label == "bus" and id not in bus_count_s:
                bus_count_s.append(id)
            elif label == "van" and id not in van_count_s:
                van_count_s.append(id)
            
    cv2.putText(frame, f"NORTHTHBOUND  car:{len(car_count_n)} truck:{len(truck_count_n)} bus:{len(bus_count_n)} bike:{len(bike_count_n)} van:{len(van_count_n)}",(50,50),font,1.5,(255,0,255),2)
    cv2.putText(frame, f"SOUTHBOUND  car:{len(car_count_s)} truck:{len(truck_count_s)} bus:{len(bus_count_s)} bike:{len(bike_count_s)} van:{len(van_count_s)}",(50,80),font,1.5,(255,0,255),2)

    client = Client(keys.account_sid, keys.auth_token)
    if msg_count <2 :
        if ((len(car_count_s) + len(truck_count_s) + len(bus_count_s) + len(bike_count_s) + len(van_count_s)) > 10 and s_alert ==False) :
            msg_count += 1 
            s_alert = True
            message = client.messages.create(
            body = "ALERT! TRAFFIC ON SOTHBOUND",
            from_=keys.twilio_number,
            to=keys.target_number
            )
        elif ((len(car_count_n) + len(truck_count_n) + len(bus_count_n) + len(bike_count_n) + len(van_count_n)) > 10 and n_alert ==False):
            msg_count += 1
            n_alert = True
            message = client.messages.create(
            body = "ALERT! TRAFFIC ON NORTHBOUND",
            from_=keys.twilio_number,
            to=keys.target_number
            ) 
    # Display frame
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    frame = cv2.resize(frame,(900,700)) #Display Window Size
    cv2.imshow("HI", frame) 
    size = (900,700)
    
    
    # Exit when 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
