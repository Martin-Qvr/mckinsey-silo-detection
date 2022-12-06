import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

from torch import nn

class SegmentationModel(nn.Module):

  def __init__(self, ENCODER, WEIGHTS):
    super(SegmentationModel, self).__init__()

    self.backbone = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = WEIGHTS,
        in_channels = 3,
        classes = 1,
        activation = None)
    
  def forward(self, images, masks=None):
    logits = self.backbone(images)

    if masks != None:

      return logits, DiceLoss(mode="binary")(logits, masks) + nn.BCEWithLogitsLoss()(logits, masks) # We use both the Diceloss and the Binary crossentropy

    return logits # During the predictions mask is going to be None, so we only return the predictions