import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision
import segmentation_models_pytorch as smp

from torchvision import transforms
from torch import nn
from segmentation_models_pytorch.losses import DiceLoss

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


long_df = px.data.medals_long()

fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")


class PretrainedModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.effnet_model = torchvision.models.efficientnet_b0(pretrained=True)
        
        # for param in effnet_model.features.parameters():
        #     param.requires_grad = False
        self.effnet_model.features[0] = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.SiLU(inplace=True)
                    )
        self.effnet_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), 
            
            nn.Linear(in_features=1280, 
                        out_features=1, # same number of output units as our number of classes
                        bias=True))
    
    def forward(self, x):
        return self.effnet_model(x)



class SegmentationModel(nn.Module):

  def __init__(self):
    super(SegmentationModel, self).__init__()

    self.backbone = smp.Unet(
        encoder_name = "timm-efficientnet-b0",
        encoder_weights = "imagenet",
        in_channels = 3,
        classes = 1,
        activation = None)
    
  def forward(self, images, masks=None):
    logits = self.backbone(images)

    if masks != None:

      return logits, DiceLoss(mode="binary")(logits, masks) + nn.BCEWithLogitsLoss()(logits, masks) # We use both the Diceloss and the Binary crossentropy

    return logits # During the predictions mask is going to be None, so we only return the predictions
    
with st.container():
    st.subheader("Hello! We are Group 4, the Musketeers :wave:")
    st.title(":sunglasses:Silo detection in images🛰️")

classification_model = PretrainedModel()
dict_model_class = torch.load('pretrained.pth', map_location=torch.device('cpu'))
classification_model.load_state_dict(dict_model_class)
classification_model.eval()

segmentation_model = SegmentationModel()
dict_model_seg = torch.load('second_model.pt', map_location=torch.device('cpu'))
segmentation_model.load_state_dict(dict_model_seg)
segmentation_model.eval()


with st.container():
    st.write("---")
    st.header("Classification")
    st.write("Let's browse some images and see if we can spot all the silos :wink::")

    picture_list = st.file_uploader(label="Choose a bunch of satellite pictures",
                               type=".png",
                               accept_multiple_files=True,
                               label_visibility="hidden")
    
    list_with = []
    list_without = []
    
    with torch.no_grad():
        for picture in picture_list:
            nparr = np.fromstring(picture.read(), np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            trans = transforms.Compose([transforms.ToTensor()])
            img_tensor = trans(img_np)
            with_silo = torch.round(torch.sigmoid(classification_model(img_tensor[None, :, :, :]))).squeeze()
            if with_silo >= 0.5:
                mask = torch.sigmoid(segmentation_model(img_tensor[None, :, :, :]))
                mask = (mask > 0.5) * 1
                mask = mask[0, :, :, :].permute(1,2,0).squeeze().numpy()
                img = (np.floor(img_np * (mask[:,:, None] + (mask[:,:, None] < 1) * 0.4))).astype(np.uint8)
                list_with.append(img)
            else:
                list_without.append(picture)
            
    expander_with_silo = st.expander(label="Pictures with Silo")
    expander_with_silo.image(list_with)
    expander_without_silo = st.expander(label="Pictures without Silo")
    expander_without_silo.image(list_without)
    if picture_list:
        bar_df = pd.DataFrame({"Number of pictures": [len(list_with), len(list_without)],
                               " ": ["Pictures with silos", "Picture without silos"]})
        fig = px.bar(bar_df, x=" ", y="Number of pictures",
                     color_discrete_sequence=["#031119", "#3EAAF4"],
                     title="Number of pictures with and without silos")
        st.plotly_chart(fig)
        
with st.container():
    st.write("---")
    st.header("FoodX impact on the world")
    st.write("Let's pick a number of silos and see what we obtain:")
    #@st.cache(suppress_st_warning=True)
    def slider_func():
        a = st.slider('Number', 1, 1000, 100)
        return a
    
    a = slider_func()
    st.write("Installing <font color='red'>{}</font> silos could generate around € <font color='red'>{}</font> M of revenues but most importantly save from famine around <font color='red'>{}</font> k human lives!".format(a, a * 729000 *1.15 //1000000,  a*2),
             unsafe_allow_html=True)
    
