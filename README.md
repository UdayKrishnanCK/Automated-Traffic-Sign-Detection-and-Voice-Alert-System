# Automated-Traffic-Sign-Detection-and-Voice-Alert-System

## Overview

This project is an automated system for detecting traffic signs from real-time video feeds captured by a webcam. The system leverages two machine learning models: 

1. **YOLO (You Only Look Once) model** for detecting traffic signs in video frames.
2. **CNN (Convolutional Neural Network) model** for classifying the detected traffic signs into one of 43 categories.

Upon detection and classification of a traffic sign, the system triggers a voice alert corresponding to the identified sign, enhancing driving safety and awareness.

## Project Structure

- **`models/`**: This directory contains the trained models used in the project.
  - **`best.pt`**: The YOLO model trained specifically for detecting traffic signs in video frames. This model scans each frame and identifies regions that likely contain traffic signs.
  - **`best_model_new.keras`**: The CNN model trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset. Once a sign is detected by the YOLO model, this CNN model classifies it into one of the 43 
    predefined traffic sign categories.
 
    
    *Note: Due to the large size of this file, it cannot be uploaded directly to GitHub. Instead, you can download it from the following Google Drive link:*
    [https://drive.google.com/file/d/1Hd95K9b97MoocIFfzg454uiN3-9eh90i/view?usp=sharing]

- **`notebooks/`**: This directory contains the Jupyter notebook that includes the full codebase.
  - **`Automated Traffic Sign Detection and Voice Alert System1.ipynb`**: The notebook where all the code is implemented, including model loading, video capture, traffic sign detection, classification, and voice alert generation.

- **`data/`**: This directory holds the dataset and associated files.
  - **`Dataset/`**: Contains subfolders for each of the 43 traffic sign classes. Each subfolder contains images belonging to that class, used for training the CNN model.


    *Note: The dataset is large, so it has been uploaded to Google Drive instead of GitHub. You can download it from the following link:*
    [https://drive.google.com/drive/folders/1bcP1BHyo-dIH5_VAhOncbk8AaSgS3RBH?usp=sharing]
  - **`data.npy`**: A numpy array storing the input data, typically preprocessed images, which were used for training the models.
 

      *Note: This file is also too large to be uploaded to GitHub and is available for download via Google Drive:*
     [https://drive.google.com/file/d/1nRlQy9Xy4B3ETLCQ77b56EIrLdlpzdzM/view?usp=sharing]
  - **`target.npy`**: A numpy array storing the target labels corresponding to the input data, used for supervised learning in model training.

- **`.gitignore`**: A file specifying which files and directories to ignore when committing to the repository. This typically includes temporary files, logs, and model weights that can be regenerated.

- **`README.md`**: The project documentation file you are currently reading. It provides a comprehensive overview of the project, its structure, and how to use it.

## Installation

### Prerequisites

Ensure you have Python 3.7+ installed on your system. You will also need `pip`, the Python package manager, to install the required libraries.

### Steps

1. **Clone the Repository**:
   - Clone this repository to your local machine using the following command:
     ```
     git clone https://github.com/UdayKrishnanCK/Automated-Traffic-Sign-Detection-and-Voice-Alert-System.git
     cd Automated-Traffic-Sign-Detection-and-Voice-Alert-System
     ```

2. **Install Dependencies**:
   - Install the necessary Python packages using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

   The `requirements.txt` file includes essential libraries such as TensorFlow, OpenCV, numpy, and others required for running the models and processing the video feed.

## Usage

### Running the System

1. **Open the Jupyter Notebook**:
   - Launch Jupyter Notebook and open `Automated Traffic Sign Detection and Voice Alert System1.ipynb` located in the `notebooks/` directory.

2. **Execute the Notebook**:
   - Follow the cells sequentially to:
     - Load the pre-trained YOLO and CNN models.
     - Capture video from your webcam.
     - Process the video frames to detect traffic signs.
     - Classify the detected signs and trigger voice alerts.
   - Ensure your webcam is connected and accessible by your system.

### System Workflow

1. **Video Capture**:
   - The system captures live video frames from the webcam.

2. **Traffic Sign Detection**:
   - The YOLO model processes each frame to detect regions that likely contain traffic signs.

3. **Traffic Sign Classification**:
   - Detected regions are passed to the CNN model, which classifies them into one of the 43 traffic sign classes.

4. **Voice Alert**:
   - Once a sign is classified, the system generates a voice alert corresponding to the detected traffic sign.

### Example Use Cases

- **Car Dashboard Camera**: Integrate the traffic sign detection system with a car's dashboard camera to enhance driver safety. The system can continuously monitor the road ahead, detect traffic signs, and provide real-time voice alerts to the driver, helping them stay informed about important road signs without needing to take their eyes off the road
- **Educational Tools**: Employ the system in educational settings to demonstrate and teach about traffic signs and their importance, using real-time detection and alerts for interactive learning
- **Research and Development**: Use this project as a foundation for developing more advanced traffic sign recognition systems or integrating additional features such as speed limit enforcement.

## Dataset

The dataset used for training the CNN model is based on the **German Traffic Sign Recognition Benchmark (GTSRB)**, which contains images of 43 different traffic sign classes. Each class represents a unique type of traffic sign, such as speed limits, no entry, stop signs, etc.

The dataset is organized into subfolders within the `data/Dataset/` directory, with each subfolder named after its respective class and containing the relevant images.

## Models

- **YOLO Model (`best.pt`)**:
  - **Purpose**: Detects traffic signs in real-time video frames.
  - **Training**: Trained on a custom dataset containing traffic signs. The model was fine-tuned to optimize detection accuracy, particularly for smaller and more complex signs.

- **CNN Model (`best_model_new.keras`)**:
  - **Purpose**: Classifies the detected traffic signs into one of 43 categories.
  - **Training**: Trained on the GTSRB dataset, achieving high accuracy of 99.47%. The model was saved in the `.keras` format, which is compatible with TensorFlow and Keras for easy deployment.

## Acknowledgements

- **GTSRB Dataset**: The German Traffic Sign Recognition Benchmark was instrumental in training the CNN model. 
- **YOLO Framework**: The YOLO architecture provided an efficient way to detect objects in real-time with minimal computational overhead.
- **OpenCV and TensorFlow**: These libraries were critical for video processing and model deployment.
