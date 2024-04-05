# ERAV2-Session-10
## Problem Statement

1. Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:  
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]  
    2. Layer1 -  
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]  
        2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        3. Add(X, R1)  
    3. Layer 2 -  
        1. Conv 3x3 [256k]  
        2. MaxPooling2D  
        3. BN  
        4. ReLU  
    4. Layer 3 -  
        1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]  
        2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]  
        3. Add(X, R2)  
    5. MaxPooling with Kernel Size 4  
    6. FC Layer  
    7. SoftMax 
2. Uses One Cycle Policy such that:  
    1. Total Epochs = 24  
    2. Max at Epoch = 5  
    3. LRMIN = FIND  
    4. LRMAX = FIND  
    5. NO Annihilation. 
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)  
4. Batch size = 512  
5. Use ADAM, and CrossEntropyLoss  
6. Target Accuracy: 90%  
7. NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training.  
8. Once done, proceed to answer the Assignment-Solution page.  

## Files Overview:
1. data_loader.py
This file defines classes and functions for loading, transforming, and summarizing the CIFAR-10 dataset using PyTorch and Albumentations library for data augmentation, normalization, and visualization.
2. custom_resnet.py
This file defines our custom resnet model.
3. main.py
This file defines classes and functions for training and testing our model.
4. S10.ipynb
`Session_10.ipynb` is a Jupyter Notebook file where the main code is written and experiment is conducted.

## Instructions:

To run the code:

1. Clone the repository to your local machine: 
2. Navigate to the project directory:
cd ERAV2-Session-10
3. Open `Session_10.ipynb` using Jupyter Notebook:
4. Follow the instructions in the notebook to train the model, evaluate its performance, and make predictions on new data.

### Additional Notes:

- Make sure you have Python and Jupyter Notebook installed on your system.
- The project uses PyTorch for implementing the CNN model.
- For any questions or issues, please contact [Utkarsh Gupta](mailto:utkarsh2198@gmail.com).
