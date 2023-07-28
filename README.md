# Sign Language Classifier

![Sign Language Classifier](sign_language.png)

This repository contains the code for a Sign Language Classifier. The classifier is built using deep learning techniques to identify different sign language gestures and classify them into corresponding letters.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Image](#sample-image)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Sign language is a powerful means of communication for individuals with hearing impairment. This Sign Language Classifier can recognize various sign language gestures and predict the corresponding letters, helping to bridge the communication gap.

## Features

- Upload an image: Users can upload an image containing a sign language gesture for classification.
- Real-time Classification: Users can use the webcam to make a sign, and the model will provide real-time predictions.

## Installation

1. Clone the repository: git clone https://github.com/Graceemeruwa/SIGN_LANGUAGE_CLASSIFIER.git
cd SIGN_LANGUAGE_CLASSIFIER


2. Install the required libraries:
pip install -r requirements.txt

3. Download the trained model:
- Download the trained model file `sign_lang_model.hdf5` from the link provided in the repository.
- Place the `sign_lang_model.hdf5` file in the root directory of the project.

## Usage

1. Run the Streamlit app:
streamlit run app.py

2. Choose an option:
- Select "Upload Image" to choose an image from your local system for classification.

4. Image Upload:
- If you choose "Upload Image," use the "Choose an image..." button to select an image from your local system.
- The app will display the uploaded image and provide the predicted sign language gesture.

## Sample Image

![Sample Image](sign_language_gestures.PNG)

This is a sample image showing all the sign language gestures that the classifier can recognize.

## License

This project is licensed under the [MIT License](LICENSE).
