from Gesture import Gesture
from Capture import Capture
from Client import Client
import cv2
import matplotlib.pyplot as plt

gesture_rec = Gesture("modelv4.h5")
camera = Capture()
client = Client('127.0.0.1', 12345)
data = client.get_data()

while True:
    try:
        camera.show_camera()
        key = cv2.waitKey(1)

        print(data)

        if data == "Waiting for Opponent":
            #camera.write_msg("Waiting for Opponent")
            data = client.get_data()
        elif data == "Make Move":
            continue
            #start timer for hand gesture
        elif data == "Winner":
            pass
            #camera.write_msg("Winner")
        elif data == "Loser":
            pass
            #camera.write_msg("Loser")
        else:
            output = 'Error: {}'.format(data)

        if key == ord('s'):
            image = camera.capture_hand()
            prediction = gesture_rec.predict(image)

            print(prediction)

            break
        elif key == ord('q'):
            camera.shutdown()
            break

    except KeyboardInterrupt:
        camera.shutdown()
        break
