import numpy as np
import cv2
import random


class RandomMisalignmentOverlap(object):
    def __init__(self, shift_x=1, shift_y=1, alpha=0.65, probability=0.5):
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.alpha = alpha
        self.probability = probability

    def __call__(self, origin_image):
        if random.random() < self.probability:
            blended_image = []
            for z_index in range(len(origin_image)):
                num_y, num_x = origin_image[z_index].shape[:2]
                M = np.float32([[1, 0, self.shift_x], [0, 1, self.shift_y]])
                translated_image = cv2.warpAffine(origin_image[z_index], M, (num_y, num_x))
                # print(translated_image.shape, origin_image.shape)
                blended_image.append(cv2.addWeighted(origin_image[z_index], self.alpha, translated_image, 1 - self.alpha, 0))

            return np.array(blended_image)
        else: return origin_image
