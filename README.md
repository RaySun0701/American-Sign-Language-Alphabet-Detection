# American Sign Language Alphabet Detection

**Dataset Source:**
 The dataset used in this project can be downloaded from Kaggle:
[Lexset Synthetic ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data)

> This dataset created by **Lexset** contains **27,000 images** of the alphabet signed in **American Sign Language (ASL)**.
>  Each image is **512 × 512 pixels**.
>  The data is separated into a **training set** and a **testing set**.
>  Within each set, there are **27 folders** — one for each letter (A–Z) and one extra folder of random backgrounds.
>  Each training folder contains **900 examples**, while each testing folder contains **100 examples**.

------

## Project Overview

This project is a simple **Flask web application** that detects and classifies **American Sign Language (ASL) hand signs** from uploaded images using a trained **Convolutional Neural Network (CNN)** model.

- The trained model (`ASL.h5`) was built using TensorFlow/Keras.
- Users can upload an image through the web interface to get the predicted ASL letter.
- The backend handles preprocessing using OpenCV, and inference with TensorFlow.

------

## Project Structure

```
AMERICAN_SIGN_LANGUAGE_DETECTION/
│
├── .venv/                       # Virtual environment (not included in GitHub)
│
├── app/
│   ├── static/                  # Static files (uploaded images, etc.)
│   ├── templates/               # HTML templates (index.html)
│   ├── app.py                   # Flask web application
│   ├── model.py                 # Model loading and preprocessing logic
│   └── ASL.h5                   # Trained CNN model used by Flask
│
├── images_from_test_set/        # Sample images from the test set for app testing
│
├── ML_model/
│   ├── ASL.ipynb                # Data reading, preprocessing, and model training notebook
│   └── ASL.h5                   # Trained model output from the notebook
│
└── requirements.txt             # Dependencies for setting up the environment
```

------

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/American_Sign_Language_Detection.git
cd American_Sign_Language_Detection
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

------

## Running the Flask App

Start the app with:

```bash
python app/app.py
```

Once started, the terminal will display something like:

```
 * Running on http://127.0.0.1:5000
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

Upload an image (for example, from the `images_from_test_set` folder) and the app will return the predicted ASL letter.

------

## Port and Network Settings

- **Default local address:**
   `http://127.0.0.1:5000`

- **If port 5000 is already in use or you want a different port:**

  - CLI command:

    ```bash
    flask run --port 5001
    ```

  - Or modify your code:

    ```python
    app.run(debug=True, port=5001)
    ```

- **To make the app accessible to other devices on the same network:**

  ```python
  app.run(host="0.0.0.0", port=5000)
  ```

  > ⚠️ Note: Make sure your firewall allows inbound connections on the selected port.

------

## Model Training

To understand how the CNN model was trained:

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook ML_model/ASL.ipynb
   ```

2. The notebook covers:

   - Dataset loading and visualization
   - Image preprocessing
   - CNN architecture design
   - Model training and evaluation
   - Saving the trained model (`ASL.h5`)


------

## License

This project is for **educational and research purposes** only.
 All rights to the dataset belong to **Lexset**.
 Please refer to the [Kaggle dataset license](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data) for usage terms.
