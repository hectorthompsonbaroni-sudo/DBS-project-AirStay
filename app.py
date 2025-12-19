from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import random
import os
from dotenv import load_dotenv
from supabase import create_client
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
MODEL_DIR = './models'

kmeans_model = None
scaler = None
feature_names = None
segment_profiles = None
supabase_client = None

hotel_price_model = None
flight_price_model = None

def load_model():
    global kmeans_model, scaler, feature_names, segment_profiles
    try:
        kmeans_model = joblib.load(f'{MODEL_DIR}/kmeans_model.pkl')
        scaler = joblib.load(f'{MODEL_DIR}/scaler.pkl')
        feature_names = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
        segment_profiles = joblib.load(f'{MODEL_DIR}/segment_profiles.pkl')
        return True
    except Exception:
        return False

def load_price_models():
    global hotel_price_model, flight_price_model
    try:
        try:
            hotel_price_model = joblib.load(f'{MODEL_DIR}/hotel_price_model.pkl')
        except Exception:
            hotel_price_model = None
        try:
            flight_price_model = joblib.load(f'{MODEL_DIR}/flight_price_model.pkl')
        except Exception:
            flight_price_model = None
        return True
    except Exception:
        return False

def predict_hotel_price(hotel_row):
    if hotel_price_model is None:
        return None
    try:
        df_input = pd.DataFrame([{
            "Location": hotel_row.get('city', 'Unknown'),
            "Rating": float(hotel_row.get('hotelrating', 3.0)),
            "Room Type": hotel_row.get('roomtype', 'Standard'),
            "Bed Type": hotel_row.get('bedtype', 'Double'),
        }])
        return float(hotel_price_model.predict(df_input)[0])
    except Exception:
        return None

def predict_flight_price(flight_row):
    if flight_price_model is None:
        return None
    try:
        date_col = None
        for col in ['date', 'departuredate', 'departure_date', 'flightdate']:
            if col in flight_row:
                date_col = col
                break
        if date_col is None:
            return None
        d = pd.to_datetime(flight_row[date_col])
        df_input = pd.DataFrame([{
            "Airline": str(flight_row.get('airlineid', '1')),
            "Source": str(flight_row.get('departureairportid', '1')),
            "Destination": str(flight_row.get('arrivalairportid', '1')),
            "Journey_month": int(d.month),
            "Journey_day": int(d.day),
        }])
        return float(flight_price_model.predict(df_input)[0])
    except Exception:
        return None

def connect_supabase():
    global supabase_client
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return True
    except Exception:
        return False

def get_customer_features(customer_id):
    try:
        customer_response = supabase_client.table('customers').select('*').eq('customerid', customer_id).execute()
        if not customer_response.data:
            return None, "Customer not found"
        customer_info = customer_response.data[0]
        dob = pd.to_datetime(customer_info['dob'])
        age = (datetime.now() - dob).days // 365
        activity_response = supabase_client.table('customer_activity').select('*').eq('customerid', customer_id).execute()
        activity_data = activity_response.data
        num_page_views = sum(1 for a in activity_data if a['type'] == 'page_view')
        num_searches = sum(1 for a in activity_data if a['type'] == 'search')
        if activity_data:
            dates = set(pd.to_datetime(a['timestamp']).date() for a in activity_data)
            days_active = len(dates)
        else:
            days_active = 0
        avg_session_duration = num_page_views * 0.5 if num_page_views > 0 else 0
        transactions_response = supabase_client.table('transaction').select('*').eq('customerid', customer_id).execute()
        transactions = transactions_response.data
        num_purchases = len(transactions)
        avg_previous_price = np.mean([t['amount'] for t in transactions]) if transactions else 0
        features = {
            'CustomerAge': age,
            'TotalPageViews': num_page_views,
            'TotalSearches': num_searches,
            'DaysActiveLast30': days_active,
            'AvgSessionDuration': avg_session_duration,
            'PreviousPurchaseCount': num_purchases
        }
        if 'AvgPreviousPackagePrice' in feature_names:
            features['AvgPreviousPackagePrice'] = avg_previous_price
        return features, None
    except Exception as e:
        return None, str(e)

def predict_segment(features):
    X = np.array([[features[fname] for fname in feature_names]])
    X_scaled = scaler.transform(X)
    segment = kmeans_model.predict(X_scaled)[0]
    return segment

def generate_packages_for_customer(segment, num_packages=5):
    try:
        segment_profile = segment_profiles[segment]
        packages_response = supabase_client.table('package').select('country').execute()
        packages_df = pd.DataFrame(packages_response.data)
        hotels_response = supabase_client.table('hotel').select('*').execute()
        hotels_df = pd.DataFrame(hotels_response.data)
        flights_response = supabase_client.table('flight').select('*').execute()
        flights_df = pd.DataFrame(flights_response.data)
        airports_response = supabase_client.table('airport').select('*').execute()
        airports_df = pd.DataFrame(airports_response.data)
        cars_response = supabase_client.table('car').select('*').execute()
        cars_df = pd.DataFrame(cars_response.data)
        popular_countries = packages_df['country'].value_counts().head(5).index.tolist()
        flight_date_col = None
        for col in ['departuredate', 'departure_date', 'flightdate', 'date']:
            if col in flights_df.columns:
                flight_date_col = col
                break
        generated_packages = []
        for i in range(num_packages):
            destination_country = random.choice(popular_countries)
            available_hotels = hotels_df[hotels_df['country'] == destination_country]
            if len(available_hotels) == 0:
                available_hotels = hotels_df
            if segment_profile.get('avg_package_price', 0) > 2000:
                high_rated = available_hotels[available_hotels['hotelrating'] >= 4]
                hotel = high_rated.sample(1).iloc[0] if len(high_rated) > 0 else available_hotels.sample(1).iloc[0]
            else:
                lower_rated = available_hotels[available_hotels['hotelrating'] <= 3.5]
                hotel = lower_rated.sample(1).iloc[0] if len(lower_rated) > 0 else available_hotels.sample(1).iloc[0]
            destination_airports = airports_df[airports_df['country'] == destination_country]
            if len(destination_airports) > 0:
                dest_airport = destination_airports.sample(1).iloc[0]['airportid']
                outbound_flights = flights_df[flights_df['arrivalairportid'] == dest_airport]
                inbound_flights = flights_df[flights_df['departureairportid'] == dest_airport]
                outbound_flight = outbound_flights.sample(1).iloc[0] if len(outbound_flights) > 0 else flights_df.sample(1).iloc[0]
                inbound_flight = inbound_flights.sample(1).iloc[0] if len(inbound_flights) > 0 else flights_df.sample(1).iloc[0]
            else:
                outbound_flight = flights_df.sample(1).iloc[0]
                inbound_flight = flights_df.sample(1).iloc[0]
            if flight_date_col:
                outbound_date = pd.to_datetime(outbound_flight[flight_date_col])
                inbound_date = pd.to_datetime(inbound_flight[flight_date_col])
                trip_duration = max(1, (inbound_date - outbound_date).days)
            else:
                from datetime import timedelta
                outbound_date = datetime.now() + timedelta(days=random.randint(7, 14))
                trip_duration = random.randint(5, 10)
                inbound_date = outbound_date + timedelta(days=trip_duration)
            car_info = None
            car_price = 0
            if random.random() < 0.3 and len(cars_df) > 0:
                car = cars_df.sample(1).iloc[0]
                car_info = {
                    'license': car['licenseno'],
                    'make': car.get('make', 'N/A'),
                    'model': car.get('model', 'N/A')
                }
                car_price = car['price']
            hotel_price = float(hotel['baseprice']) * trip_duration
            flight_price = float(outbound_flight['baseprice']) + float(inbound_flight['baseprice'])
            total_car_price = float(car_price) * trip_duration if car_info else 0
            total_price = int(hotel_price + flight_price + total_car_price)
            package = {
                'destination': destination_country,
                'hotel': {
                    'name': hotel.get('hotelname', 'Hotel'),
                    'rating': float(hotel['hotelrating']),
                    'price_per_night': float(hotel['baseprice'])
                },
                'outbound_flight': {
                    'number': outbound_flight['flightnumber'],
                    'date': outbound_date.strftime('%Y-%m-%d'),
                    'price': float(outbound_flight['baseprice'])
                },
                'inbound_flight': {
                    'number': inbound_flight['flightnumber'],
                    'date': inbound_date.strftime('%Y-%m-%d'),
                    'price': float(inbound_flight['baseprice'])
                },
                'car': car_info,
                'trip_duration': trip_duration,
                'total_price': total_price
            }
            generated_packages.append(package)
        return generated_packages, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        customer_id = data.get('customer_id')
        num_packages = data.get('num_packages', 5)
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400
        features, error = get_customer_features(customer_id)
        if error:
            return jsonify({'error': error}), 404
        segment = predict_segment(features)
        segment_info = segment_profiles[segment]
        packages, error = generate_packages_for_customer(customer_id, segment, num_packages)
        if error:
            return jsonify({'error': error}), 500
        return jsonify({
            'customer_id': customer_id,
            'features': features,
            'segment': {
                'id': int(segment),
                'name': segment_info['name'],
                'avg_age': segment_info['avg_age'],
                'avg_previous_purchases': segment_info['avg_previous_purchases']
            },
            'packages': packages
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': kmeans_model is not None,
        'supabase_connected': supabase_client is not None,
        'num_segments': len(segment_profiles) if segment_profiles else 0
    })

@app.route('/api/flights')
def get_flights():
    try:
        response = supabase_client.table('flight').select('*').execute()
        flights = response.data
        if flight_price_model is not None:
            for flight in flights:
                predicted_price = predict_flight_price(flight)
                flight['predicted_price'] = predicted_price // 100
        return jsonify({'flights': flights})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hotels')
def get_hotels():
    try:
        response = supabase_client.table('hotel').select('*').execute()
        hotels = response.data
        if hotel_price_model is not None:
            for hotel in hotels:
                predicted_price = predict_hotel_price(hotel)
                hotel['predicted_price'] = predicted_price // 1000
        return jsonify({'hotels': hotels})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not load_model():
        exit(1)
    load_price_models()
    if not connect_supabase():
        exit(1)
    app.run(debug=True, host='0.0.0.0', port=5050)
