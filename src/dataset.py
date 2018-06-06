from skimage import color, io, data, transform
import numpy as np
import os


class DataSet:  # read the image and generate training data
    def __init__(self, directory):
        self.files = []
        self.directory = directory
        for file in os.listdir(directory):
            self.files.append(file)
        np.random.shuffle(self.files)

    def gen_corrupted_image(self, image, noise_ratio=0.8): #  generate the corrupted image
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        corrupted_image = np.copy(image)
        rows, cols, channels = corrupted_image.shape
        noise_number = int(np.round(noise_ratio * cols))  #  get the number of noisy pixels 
        for channel in range(channels):
            for row in range(rows):
                tmp = np.random.permutation(cols)[0:noise_number]
                corrupted_image[row, tmp, channel] = 0
        return corrupted_image

    def get_data(self, channels=3):  # generate training data
        corrupted_images = []
        residual_images = []
        for i, file in enumerate(self.files, 1):
            print('%d / %d' % (i, len(self.files)))
            try:
                image = io.imread(os.path.join(self.directory, file))
                if channels == 1:
                    image = color.rgb2gray(image)
            except BaseException as e:
                print('error', e)
                continue
            if len(image.shape) <= 1:
                continue
            if channels == 3:
                image = transform.resize(image, [240, 240, 3])
            else:
                image = transform.resize(image, [240, 240])
            for _ in range(4):  #  reinforce training data
                corrupted_image = self.gen_corrupted_image(image)
                corrupted_images.append(corrupted_image)
                if channels == 3:
                    residual_images.append(image)
                else:
                    residual_images.append(np.expand_dims(image, axis=2))
                if i != 3:
                    image = transform.rotate(image, 90)
            if len(corrupted_images) >= 100:
                yield (np.array(corrupted_images), np.array(residual_images))
                corrupted_images = []
                residual_images = []
        yield (np.array(corrupted_images), np.array(residual_images))