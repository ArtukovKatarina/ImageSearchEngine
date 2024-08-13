from torch import nn
from image_search_engine.src import config as CFG
import timm

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.MODEL_NAME, pretrained=CFG.PRETRAINED, trainable=CFG.TRAINABLE
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)