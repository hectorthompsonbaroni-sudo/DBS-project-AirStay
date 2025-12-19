# AirStay

A machine learning-powered travel package recommendation system that generates personalized hotel, flight, and car rental packages based on customer segmentation and behavioral analysis.

## Project Overview

AirStay uses KMeans clustering to segment customers based on their demographics and browsing behavior, then generates tailored travel package recommendations. The system also includes dynamic pricing models for hotels and flights.

## File Structure

### Core Application Files

- **app.py** - Main Flask application that handles:
  - Customer segmentation using ML models
  - Personalized package generation
  - Hotel and flight price prediction
  - REST API endpoints for the frontend
  - Supabase database integration

### Machine Learning & Data

- **create_package_model.ipynb** - Jupyter notebook for training the customer segmentation model (KMeans clustering)
- **Dynamic Price Prediction.ipynb** - Jupyter notebook for training hotel and flight price prediction models
- **PACKAGE_TRAINING_DATA.csv** - Training dataset for the package recommendation model

### Frontend

- **templates/index.html** - Web interface for interacting with the recommendation system

### Configuration

- **.gitignore** - Git ignore rules for the project

## Setup Guide

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DBS-project-AirStay
   ```

2. **Install dependencies**
   ```bash
   pip install flask pandas numpy joblib python-dotenv supabase scikit-learn jupyter
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

4. **Train the models** (if models directory doesn't exist)

   Run the Jupyter notebooks to generate the required model files:
   - Open `create_package_model.ipynb` and run all cells to create customer segmentation models
   - Open `Dynamic Price Prediction.ipynb` and run all cells to create price prediction models

   This will create a `models/` directory with:
   - `kmeans_model.pkl`
   - `scaler.pkl`
   - `feature_names.pkl`
   - `segment_profiles.pkl`
   - `hotel_price_model.pkl`
   - `flight_price_model.pkl`

5. **Run the application**
   ```bash
   python app.py
   ```

   The application will start on `http://localhost:5050`

### Database Requirements

The application expects the following tables in your Supabase database:
- `customers` - Customer information (customerid, dob, etc.)
- `customer_activity` - Customer browsing activity (type: page_view, search)
- `transaction` - Purchase history (customerid, amount)
- `package` - Available travel packages
- `hotel` - Hotel inventory
- `flight` - Flight inventory
- `airport` - Airport information
- `car` - Car rental inventory

## API Endpoints

- `GET /` - Web interface
- `POST /api/generate` - Generate personalized packages for a customer
- `GET /api/status` - Check model and database connection status
- `GET /api/flights` - Get all flights with predicted prices
- `GET /api/hotels` - Get all hotels with predicted prices

## Usage

1. Navigate to `http://localhost:5050` in your browser
2. Enter a customer ID to generate personalized travel package recommendations
3. The system will analyze the customer's profile and browsing behavior
4. View recommended packages tailored to the customer's segment
