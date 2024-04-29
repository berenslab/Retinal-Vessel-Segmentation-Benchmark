
import segmentation_models_pytorch as smp


class MANet(smp.MAnet):
    def __init__(self):
        super(MANet, self).__init__()
        self.model = smp.MAnet(encoder_name='resnet18',
                               in_channels=3, classes=1, activation=None)

    def forward(self, x):
        return self.model.forward(x)
