import queue
import threading

import cv2
from Capture import Capture
from Client import Client
from Gesture import Gesture

que = queue.Queue()
gesture_rec = Gesture("modelv13.h5")
camera = Capture()
client = Client('127.0.0.1', 12345)
data = {'status':''}
client.get_data(que)
frame_count = 0
display_msg = False
display_time = False
display_info = False
time = 0
bullets = 0
rounds = 0
player_move = ''
opp_move = ''
msg = ""
stat_data = dict()

if not que.empty():
    data = que.get()

while True:
    try:
        if not que.empty():
            data = que.get()
            if display_info:
                rounds = data["Round"]
                bullets = data["Bullet Count"]
                player_move = data["Player Move"]
                opp_move = data["Opp Move"]

        key = cv2.waitKey(1)

        camera.show_camera(display_msg, msg, display_time, time, display_info, rounds, bullets, player_move, opp_move,
                           stat_data)
        if data['status'] == "Waiting for Opponent":
            display_msg = True
            msg = "Waiting for Opponent"
            display_info = True

            thread1 = threading.Thread(target=client.get_data, args=(que,))
            thread1.start()
        elif data['status'] == "Make Move":
            display_msg = True
            msg = "Make Your Move"
            display_time = True
            frame_count = 100
        elif data['status'] == "Winner":
            display_msg = True
            msg = "Winner!"
            stat_data["PStats"] = data["PStats"]
            stat_data["OppStats"] = data["OppStats"]
        elif data['status'] == "Loser":
            display_msg = True
            msg = "Loser!"
            stat_data["PStats"] = data["PStats"]
            stat_data["OppStats"] = data["OppStats"]
        elif data['status'] == "error":
            output = 'Error: {}'.format(data)

        data = {'status': ''}

        if display_time:
            time = frame_count // 10
            frame_count -= 1
            if frame_count == 0:
                display_time = False
                image = camera.capture_hand2()
                prediction = gesture_rec.predict2(image)
                print(prediction)
                msg = "Moved Sent: {}".format(prediction)

                if prediction == "RELOAD":
                    server_data = "1"
                if prediction == "SHIELD":
                    server_data = "2"
                if prediction == "SHOOT":
                    server_data = "3"

                client.send_data(server_data)
                display_msg = True

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
