### Abstract
Predicting long-term cardiac rhythm outcomes from baseline imaging remains a major challenge in atrial fibrillation (AF) management. In this study, we propose a neural network architecture, AF-Echo, based on spatio-temporal convolution and video self-attention, to predict cardiac rhythm status two years after initial diagnosis using transthoracic echocardiography (TTE) videos.

---
### Model
<img width="1208" height="341" alt="Model" src="https://github.com/user-attachments/assets/e1ffb75c-a977-499b-a588-7403dc5a3ef5" />

AF-Echo is an end-to-end spatio-temporal learning framework composed of the following components:
- **Video encoder**  
  A convolutional spatio-temporal backbone processes short echocardiography clips to extract hierarchical motion features of cardiac structures.
- **Attention-based aggregation**  
  An attention module aggregates patch-level spatio-temporal features across space and time, allowing the model to emphasize clinically relevant regions and temporal segments.
- **Prediction head**  
  The aggregated representation is mapped to the target outcome, producing a probability score for long-term AF severity (binary classification).  
  The framework optionally supports auxiliary regression heads for predicting echocardiographic functional markers to improve interpretability.

The model is trained end-to-end and supports gradient-based explainability methods for visual interpretation of predictions.

---
### Data Preprocessing
This model requires two input modalities: tabular data containing the target variables provided in a .csv file, and four-chamber (4CH) echocardiography videos provided in .avi format.
To ensure correct pairing between the two modalities, the tabular file must include dedicated identifier columns. Specifically, a video_name column should contain the base names of the corresponding video files (without the .avi extension). When clip-level processing is enabled, an additional video_id column must be provided, listing the identifiers of individual clips extracted from each video, thereby preserving a consistent mapping between tabular entries and video inputs.

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
