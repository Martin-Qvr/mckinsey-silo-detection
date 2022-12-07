# mckinsey-silo-detection

This repo is to store the code for the McKinsey silo detection project.

We are performing two tasks on a dataset of c. 2000 satellite images:

- **Classification** : does the image contains silo ? 
    - For the pedagocical exercise we built a CNN from scratch : **Accuracy : 77.5 , F1-Score : 80.3** 
    - For performance we used an EfficientNet : **Accuracy : 91.3 , F1-Score : 77.5**



- **Segmentation** : where is the silo positioned in the image ? 
    - We trained two U-net like CNN architecture and ended up with a **Dice Score of 62 %**


To install the required libraries run :
`pip install -r requirements.txt`

Finally, we built a webapp to present our results using streamlit. 
