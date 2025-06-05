import streamlit as st
from sentinelhub import SHConfig, SentinelHubRequest, BBox, CRS, DataCollection, MimeType
import cv2
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# Configure Sentinel Hub API
config = SHConfig()
config.sh_client_id = ''  # Replace with your client ID
config.sh_client_secret = ''

# Load the trained machine learning model
clf = joblib.load('deforestation_model.joblib')

# Function to fetch latitude and longitude coordinates from country name
def geocode_country(country_name):
    geolocator = Nominatim(user_agent="deforestation_app")
    location = geolocator.geocode(country_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# Function to fetch satellite image
def fetch_satellite_image(latitude, longitude):
    # Define bounding box coordinates (latitude, longitude)
    bbox = BBox([longitude - 0.1, latitude - 0.1, longitude + 0.1, latitude + 0.1], CRS.WGS84)

    # Define request parameters
    evalscript = """
        // Returns true color image (RGB) from Sentinel-2
        return [(B08 - B04) / (B08 + B04)]; 
    """
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=('2024-08-01', '2024-08-10'),
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.PNG)
        ],
        bbox=bbox,
        size=[512, 512],
        config=config
    )

    # Execute the request
    image = request.get_data()[0]  # Extract the first image from the list

    return image

# Function to extract green density
def extract_green_density(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    green_channel = hsv[:, :, 0]
    green_channel = green_channel[(green_channel >= 40) & (green_channel <= 80)]
    green_density = np.mean(green_channel)
    return green_density

# Function to detect deforestation
def detect_deforestation(image_array):
    # Extract green density feature
    green_density = extract_green_density(image_array)
    
    # Create feature matrix
    new_image_features = np.array([[green_density]])
    
    # Predict using the loaded model
    prediction = clf.predict(new_image_features)
    
    return prediction

# Function to handle click events and add markers

def main():
    st.title("Deforestation Detection")

    # Get country name from user input
    country_name = st.sidebar.text_input("Enter country name:", "United States")

    # Fetch latitude and longitude coordinates
    latitude, longitude = geocode_country(country_name)

    if latitude is not None and longitude is not None:
        st.write(f"Latitude: {latitude}, Longitude: {longitude}")

        # Fetch satellite image
        image = fetch_satellite_image(latitude, longitude)

        # Display the satellite image
        st.image(image, use_column_width=True, caption='Satellite Image')

        # Detect deforestation
        prediction = detect_deforestation(image)

        # Display the result
        if prediction == 1:
            st.error("Deforestation detected!")
        else:
            st.success("No deforestation detected.")

        
    

if __name__ == "__main__":
    main()
