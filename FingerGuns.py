from Gesture import Gesture
from Capture import Capture
from Client import Client
import threading
import cv2
import queue
import time
import matplotlib.pyplot as plt

que = queue.Queue()
gesture_rec = Gesture("modelv4.h5")
camera = Capture()
client = Client('127.0.0.1', 12345)
data = ""
client.get_data(que)
frame_count = 0
display_msg = False
display_time = False
time = 0
msg = ""

if not que.empty():
    data = que.get()

while True:
    try:
        if not que.empty():
            data = que.get()

        key = cv2.waitKey(1)

        camera.show_camera(display_msg, msg, display_time, time)

        if data == "Waiting for Opponent":
            display_msg = True
            msg = "Waiting for Opponent"
            thread1 = threading.Thread(target=client.get_data, args=(que, ))
            thread1.start()
        elif data == "Make Move":
            display_msg = True
            msg = "Make Your Move"
            display_time = True
            frame_count = 100
        elif data == "Winner":
            display_msg = True
            msg = "Winner!"
        elif data == "Loser":
            display_msg = True
            msg = "Loser!"
        elif data == "error":
            output = 'Error: {}'.format(data)

        data = ""

        if display_time:
            time = frame_count // 10
            frame_count -= 1
            if frame_count == 0:
                display_time = False
                image = camera.capture_hand()
                prediction = gesture_rec.predict(image)
                print(prediction)
                client.send_data(str(prediction[0]))
                display_msg = True
                msg = "Moved Sent: {}".format(prediction)
                thread2 = threading.Thread(target=client.get_data, args=(que,))
                thread2.start()


        # if key == ord('s'):
        #
        #     break
        if key == ord('q'):
            camera.shutdown()
            break

    except KeyboardInterrupt:
        camera.shutdown()
        break
