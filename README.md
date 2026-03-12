### Abstract
Predicting long-term cardiac rhythm outcomes from baseline imaging remains a major challenge in atrial fibrillation (AF) management. In this study, we propose a neural network architecture, AF-Echo, based on spatio-temporal convolution and video self-attention, to predict cardiac rhythm status two years after initial diagnosis using transthoracic echocardiography (TTE) videos.

---
### Installation
We recommend using a dedicated Conda environment for this model. The required dependencies can be installed using the following commands
    
    conda create -n afecho python=3.10
    conda activate afecho
    pip install -r requirements.txt

---
### Use
To use the model with your own dataset, you must first edit the configuration file located at /config/config.yaml and update the dataset paths and specifications according to your data. Once the configuration is set, the model can be run using the following command:

    python train.py --config /path/to/your/config/file

---
### Checkpoints
pretrained model provided at 
    /runs/Best-Checkpoints/best.pt
