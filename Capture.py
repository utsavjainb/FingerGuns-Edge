import cv2
import imutils
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import torch
# Apply the transformations needed
# import torchvision.transforms as T


class Capture:
    def __init__(self):
        dim = (100, 100)
        self.webcam = cv2.VideoCapture(0)
        # region of interest (ROI) coordinates
        self.top, self.right, self.bottom, self.left = 375, 625, 325, 75

        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        self.start_point = (375, 75)

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        self.end_point = (625, 325)

        # Blue color in BGR
        self.color = (255, 0, 0)

        # Line thickness of 2 px
        self.thickness = 2

        # self.dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

        self.background = './background.jpg'
        self.inventory = cv2.imread('./Inventory.png', cv2.IMREAD_UNCHANGED)
        self.inventory = cv2.resize(self.inventory, dim, interpolation=cv2.INTER_AREA)

        self.winner = cv2.imread('./Winner.png', cv2.IMREAD_UNCHANGED)
        self.winner = cv2.resize(self.winner, (200, 200), interpolation=cv2.INTER_AREA)

        self.loser = cv2.imread('./Loser.png', cv2.IMREAD_UNCHANGED)
        self.loser = cv2.resize(self.loser, (200, 200), interpolation=cv2.INTER_AREA)

        self.rounds = cv2.imread('./Rounds.png', cv2.IMREAD_UNCHANGED)
        self.rounds = cv2.resize(self.rounds, dim, interpolation=cv2.INTER_AREA)

        self.roi = None

        self.frame = None

    def show_camera(self, display_msg, msg, display_timer, time, display_info, rounds, bullets, player_move, opp_move,
                    stat_data):
        alpha = 0.3

        check, self.frame = self.webcam.read()
        self.frame = imutils.resize(self.frame, width=800)

        self.frame = cv2.flip(self.frame, 1)

        # clone the self.frame
        clone = self.frame.copy()

        # get the height and width of the self.frame
        (height, width) = self.frame.shape[:2]

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        cv2.rectangle(self.frame, self.start_point, self.end_point, self.color, self.thickness)

        # get the ROI
        self.roi = self.frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0]]
        self.roi = cv2.flip(self.roi, 1)

        if display_msg:
            if msg == "Winner!":
                cv2.putText(self.frame, msg, (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                winner_icon = cv2.add(self.frame[120:320, 75:275, :], self.winner[:, :, :3])
                self.frame[120:320, 75:275] = winner_icon

                img = self.draw_graph(stat_data)
                img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_AREA)
                graph = cv2.addWeighted(
                    self.frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0]], 0, img, 1,
                    0)
                self.frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0]] = graph


            elif msg == "Loser!":
                cv2.putText(self.frame, msg, (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                loser_icon = cv2.add(self.frame[120:320, 75:275, :], self.loser[:, :, :3])
                self.frame[120:320, 75:275] = loser_icon

                img = self.draw_graph(stat_data)
                img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_AREA)
                graph = cv2.addWeighted(
                    self.frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0]], 0, img, 1,
                    0)
                self.frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0]] = graph


            else:
                cv2.putText(self.frame, msg, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if display_timer:
            cv2.putText(self.frame, str(time), (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6, cv2.LINE_AA)

        if display_info and not (msg == "Winner!" or msg == "Loser!"):
            bullet_icon = cv2.add(self.frame[320:420, 350:450, :], self.inventory[:, :, :3])
            self.frame[320:420, 350:450] = bullet_icon

            round_icon = cv2.add(self.frame[320:420, 465:565, :], self.rounds[:, :, :3])
            self.frame[320:420, 465:565] = round_icon

            cv2.putText(self.frame, ": {}".format(bullets), (420, 375), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(self.frame, ": {}".format(rounds), (552, 375), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(self.frame, "You: {}".format(player_move), (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(self.frame, "Opp: {}".format(opp_move), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA)

        # print(check)  # prints true as long as the webcam is running
        # print(self.frame)  # prints matrix values of each self.framecd
        cv2.imshow("Capturing", self.frame)

    def draw_graph(self, data):
        colors = ["#ba6a61", "#6182ba", "#61ba7d", "#ba6a61", "#6182ba", "#61ba7d", ]
        sns.set_style("white")
        temp = dict()
        temp["Me"] = data["PStats"]
        temp["Opp"] = data["OppStats"]
        df = pd.DataFrame(temp)

        fig, ax = plt.subplots()
        df.T.plot(title="Game Statistics", ax=ax, color=sns.color_palette(colors), kind='bar', figsize=(7, 7),
                  stacked=True)
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img



    def capture_hand2(self):
        image = self.roi.copy()
        dim = (28, 28)

        img_ = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

        return img_

    def capture_hand1(self):
        cv2.imwrite(filename='saved_img.jpg', img=self.roi)
        # img_nobackground = self.remove_background(self.dlab, "./saved_img.jpg", self.background, show_orig=True)
        # cv2.imwrite(filename='saved_img_removed.jpg', img=img_nobackground)
        cv2.imwrite(filename='saved_img_removed.jpg', img=self.roi)
        # self.webcam.release()

        # img_new = cv2.imread('saved_img_removed.jpg', cv2.IMREAD_GRAYSCALE)
        # img_new = cv2.imshow("Captured Image", img_new)

        cv2.waitKey(1650)
        # cv2.destroyAllWindows()

        print("Processing image...")
        img_ = cv2.imread('saved_img_removed.jpg', cv2.IMREAD_ANYCOLOR)
        print("Converting RGB image to grayscale...")
        # gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        print("Converted RGB image to grayscale...")
        print("Resizing image to 28x28 scale...")

        print(img_.shape)

        if img_.shape[2] == 3:
            img_ = np.mean(img_, axis=-1)

        img_ = cv2.resize(img_, (28, 28))
        print("Resized...")
        cv2.imwrite(filename='saved_img-final.jpg', img=img_)
        print("Image saved!")

        return img_

    def shutdown(self):
        print("Turning off camera.")
        self.webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()

    # def remove_background(self, net, path, bgimagepath, show_orig=True, dev='cuda'):
    #     img = Image.open(path)
    #
    #     # if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
    #     # Comment the Resize and CenterCrop for better inference results
    #     trf = T.Compose([T.Resize(400),
    #                      # T.CenterCrop(224),
    #                      T.ToTensor(),
    #                      T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])])
    #     inp = trf(img).unsqueeze(0)
    #     out = net(inp)['out']
    #     om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    #
    #     rgb = self.decode_segmap(om, path, bgimagepath)
    #
    #     # plt.imshow(rgb);
    #     # plt.axis('off');
    #     # plt.show()
    #
    #     return rgb

    # Define the helper function
    def decode_segmap(self, image, source, bgimg, nc=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)

        # Load the foreground input image
        foreground = cv2.imread(source)

        # Load the background input image
        background = cv2.imread(bgimg)

        # Change the color of foreground image to RGB
        # and resize images to match shape of R-band in RGB output map
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))
        background = cv2.resize(background, (r.shape[1], r.shape[0]))

        # Convert uint8 to float
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Create a binary mask of the RGB output map using the threshold value 0
        th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

        # Apply a slight blur to the mask to soften edges
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float) / 255

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha, foreground)

        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)

        # Add the masked foreground and background
        outImage = cv2.add(foreground, background)

        # Return a normalized output image for display
        return outImage
        # return outImage / 255
