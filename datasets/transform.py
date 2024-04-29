import torch
from torchvision.transforms import functional as tf
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip



def pipeline_tranforms():
    return Compose([RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    Fix_RandomRotation(),
                    ])


class Fix_RandomRotation(object):

    def __init__(self, degrees=360, expand=False, center=None):
        self.degrees = degrees
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return tf.rotate(img, angle, expand=self.expand, center=self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        # format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string