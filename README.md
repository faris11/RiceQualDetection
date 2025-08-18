# Rice Quality Detection Application

This Streamlit application detects the quality of rice from uploaded images, classifying them into one of five categories: normal, damage, chalky, broken, or discolored. The application also validates whether the uploaded image actually contains rice.

## Features

- Upload rice images for quality analysis
- Automatic validation to check if the image contains rice
- Classification of rice quality into five categories
- Confidence score for predictions
- Descriptive information about each rice quality class

## Installation

1. Clone this repository or download the source code
2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install the required dependencies
   ```
   pip install -r requirements.txt
   ```

## Model Setup

Before running the application, you need to place your trained model in the correct location:

1. Create a directory named `model` in the project root
2. Place your trained rice quality classification model in this directory with the name `rice_quality_model.h5`

```
Rice Quality Detection/
├── app.py
├── requirements.txt
├── README.md
└── model/
    └── rice_quality_model.h5
```

## Running the Application

To run the application, execute the following command in your terminal:

```
streamlit run app.py
```

The application will start and open in your default web browser at `http://localhost:8501`.

## Usage

1. Upload an image of rice using the file uploader
2. The application will first check if the image contains rice
3. If valid, it will classify the rice quality and display the result with a confidence score
4. Additional information about rice quality classes is available in the expandable section at the bottom

## Rice Quality Classes

- **Normal**: Rice grains with standard appearance, size, and color
- **Damage**: Rice grains that have been physically damaged during harvesting or processing
- **Chalky**: Rice grains with opaque white spots due to incomplete maturation
- **Broken**: Rice grains that are not whole (less than 3/4 of a whole kernel)
- **Discolored**: Rice grains with abnormal color, often yellowish or brownish

## Requirements

- Python 3.8 or higher
- Streamlit
- TensorFlow
- OpenCV
- NumPy
- Pillow

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
