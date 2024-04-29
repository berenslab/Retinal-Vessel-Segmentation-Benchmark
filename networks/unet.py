import segmentation_models_pytorch as smp


class Unet(smp.Unet):
    def __init__(self):
        super(Unet, self).__init__()

        self.model = smp.Unet(encoder_name='resnet34',
                              in_channels=3, classes=1, activation=None)

    def forward(self, x):
        return self.model.forward(x)
   