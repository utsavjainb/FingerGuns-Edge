from keras import models
import matplotlib.pyplot as plt


class Gesture:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = models.load_model(model_name)
        print("Model Loaded")

    def predict(self, image):

        print(image.dtype)
        print(image.shape)

        # image = image / 255

        plt.imshow(image, cmap="gray")
        plt.show()

        image = image.reshape(-1, 28, 28, 1)

        predictions = self.model.predict_classes(image)

        return predictions

    def test_image(self, image):
        print(image.dtype)
        print(image.shape)
        plt.imshow(image.reshape(28, 28), cmap="gray")
        plt.show()
