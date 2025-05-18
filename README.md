# Diabetes Prediction Project

This project aims to predict whether an individual has diabetes using machine learning techniques. A web-based interface collects user input, feeds it into a trained model, and displays the prediction result on the screen.


## Project Structure
```
diabetes_project/
│
├── app.py # Main Flask application file
├── main.py # Alternative entry point file (if available)
├── requirements.txt # Required libraries
├── README.md # This file
│
├── data/
│ └── diabetes.csv # Dataset
│
├── models/
│ ├── encoder.pkl
│ ├── scaler.pkl
│ ├── voting_clf.pkl
│ └── outlier_limits.pkl # Required model components
│
├── src/
│ ├── data_preprocessing.py
│ ├── new_data_preprocess.py
│ ├── model_prediction.py
│ ├── model_training.py
│ └── utils.py # Helper functions
│
└── templates/
├── index.html # Main input form
└── predict.html # Result page

```




## Installation and Run

### 1. Create Virtual Environment
```
bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
```
### 2. Install Required Libraries
```
bash
pip install -r requirements.txt
```
### 3. Start Flask Application
```
bash
python app.py
```
### 4. Go to Web Interface

```
http://127.0.0.1:5000
```
## Model Summary
### Data preprocessing:

Filling missing values

Removing outliers

Encoding categorical variables

Normalization

### Used model:

VotingClassifier (a combination of KNN, Random Forest, LightGBM)

### Model components are saved as pickle files under the models/ folder.

## Technologies Used

| Technology  | Description                                                        |
|------------|--------------------------------------------------------------------|
| Python     | Main programming language of the project                           |
| Flask      | Micro framework used to build the web interface                    |
| Pandas     | Used for data processing and analysis                              |
| Scikit-learn | For machine learning algorithms and modeling                       |
| XGBoost    | Used for advanced classification modeling                          |
| LightGBM   | Gradient boosting algorithm used for fast and efficient modeling   |
| Pickle     | For saving/loading models and preprocessing objects                |
| JSON       | Used to save/load `num_cols` and `cat_cols` column information     |
| HTML & Jinja2 | Web pages and templating engine                                    |


