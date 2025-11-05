# Heart Disease Prediction API

A machine learning-powered REST API for predicting heart disease risk based on patient health metrics.

## Project Structure

```
.
├── train_model.py          # Model training script
├── app.py                  # Flask API application
├── test_api.py            # API testing script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── models/               # Trained model files (generated)
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   └── feature_names.json
└── heart_disease_data.csv # Training dataset
```

## Setup and Installation

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (for containerized deployment)

### Local Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare your dataset:**
Save your heart disease dataset as `heart_disease_data.csv` in the project root.

3. **Train the model:**
```bash
python train_model.py
```

This will create a `models/` directory with all necessary model artifacts.

4. **Run the API:**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## Docker Deployment

### Build and Run with Docker

1. **Build the Docker image:**
```bash
docker build -t heart-disease-api .
```

2. **Run the container:**
```bash
docker run -p 5000:5000 heart-disease-api
```

### Using Docker Compose

1. **Start the service:**
```bash
docker-compose up -d
```

2. **Stop the service:**
```bash
docker-compose down
```

3. **View logs:**
```bash
docker-compose logs -f
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### 2. Get Features
```bash
GET /features
```

Returns expected feature names and their valid values.

### 3. Make Prediction
```bash
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Age": 56.0,
  "Gender": "Male",
  "Blood Pressure": 153.0,
  "Cholesterol Level": 155.0,
  "Exercise Habits": "High",
  "Smoking": "Yes",
  "Family Heart Disease": "Yes",
  "Diabetes": "No",
  "BMI": 24.99,
  "High Blood Pressure": "Yes",
  "Low HDL Cholesterol": "Yes",
  "High LDL Cholesterol": "No",
  "Alcohol Consumption": "High",
  "Stress Level": "Medium",
  "Sleep Hours": 7.63,
  "Sugar Consumption": "Medium",
  "Triglyceride Level": 342.0,
  "Fasting Blood Sugar": 120.0,
  "CRP Level": 12.97,
  "Homocysteine Level": 12.39
}
```

**Response:**
```json
{
  "prediction": "No",
  "probability": {
    "No": 0.85,
    "Yes": 0.15
  },
  "confidence": 0.85
}
```

## Testing the API

Run the test script to verify all endpoints:

```bash
python test_api.py
```

Or test manually using curl:

```bash
# Health check
curl http://localhost:5000/health

# Get features
curl http://localhost:5000/features

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 56.0,
    "Gender": "Male",
    "Blood Pressure": 153.0,
    "Cholesterol Level": 155.0,
    "Exercise Habits": "High",
    "Smoking": "Yes",
    "Family Heart Disease": "Yes",
    "Diabetes": "No",
    "BMI": 24.99,
    "High Blood Pressure": "Yes",
    "Low HDL Cholesterol": "Yes",
    "High LDL Cholesterol": "No",
    "Alcohol Consumption": "High",
    "Stress Level": "Medium",
    "Sleep Hours": 7.63,
    "Sugar Consumption": "Medium",
    "Triglyceride Level": 342.0,
    "Fasting Blood Sugar": 120.0,
    "CRP Level": 12.97,
    "Homocysteine Level": 12.39
  }'
```

## Model Information

- **Algorithm:** Random Forest Classifier
- **Features:** 20 patient health metrics
- **Target:** Heart Disease Status (Yes/No)
- **Preprocessing:** 
  - Label encoding for categorical variables
  - Standard scaling for numerical features
  - Missing value imputation with median

## Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age in years |
| Gender | Categorical | Male/Female |
| Blood Pressure | Numeric | Systolic blood pressure |
| Cholesterol Level | Numeric | Total cholesterol |
| Exercise Habits | Categorical | Low/Medium/High |
| Smoking | Categorical | Yes/No |
| Family Heart Disease | Categorical | Yes/No |
| Diabetes | Categorical | Yes/No |
| BMI | Numeric | Body Mass Index |
| High Blood Pressure | Categorical | Yes/No |
| Low HDL Cholesterol | Categorical | Yes/No |
| High LDL Cholesterol | Categorical | Yes/No |
| Alcohol Consumption | Categorical | Low/Medium/High |
| Stress Level | Categorical | Low/Medium/High |
| Sleep Hours | Numeric | Average hours per night |
| Sugar Consumption | Categorical | Low/Medium/High |
| Triglyceride Level | Numeric | Blood triglycerides |
| Fasting Blood Sugar | Numeric | Fasting glucose level |
| CRP Level | Numeric | C-Reactive Protein level |
| Homocysteine Level | Numeric | Blood homocysteine |





