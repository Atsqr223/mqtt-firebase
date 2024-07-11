import firebase_admin
from firebase_admin import credentials, firestore
import csv
from collections import deque

# Initialize Firebase Admin SDK
cred = credentials.Certificate('C:/Users/tcs/Downloads/ball-mill-website-firebase-adminsdk-ni6zi-771a1e9b64.json')  # Update this path to your credentials file
firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# Define the maximum number of data points you want to retain
MAX_DATA_POINTS = 88

# Function to upload CSV data to Firestore and maintain the data limit
def upload_csv_to_firestore(csv_file_path):
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data_queue = deque()  # Using deque to efficiently manage FIFO queue
        
        for row in csv_reader:
            # Create a dictionary to store the data
            data = {
                "Time_Stamp1": row["Time Stamp1"],
                "Sample1_data": float(row["Sample1_data"]),
                "Time_Stamp2": row["Time Stamp2"],
                "Sample2_data": float(row["Sample2_data"]),
            }
            
            # Add the data to the queue
            data_queue.append(data)
            
            # If the queue exceeds the maximum limit, remove the oldest data
            if len(data_queue) > MAX_DATA_POINTS:
                oldest_data = data_queue.popleft()  # Remove the oldest data (FIFO)
                # Determine how to identify the document to delete (you might need a timestamp or a unique identifier)
                # In this example, assuming there's a timestamp field
                oldest_timestamp = oldest_data["Time_Stamp1"]  # Adjust based on your data structure
                # Query Firestore to find the document with the oldest timestamp and delete it
                delete_query = db.collection("samples").where("Time_Stamp1", "==", oldest_timestamp).limit(1)
                docs_to_delete = delete_query.stream()
                for doc in docs_to_delete:
                    doc.reference.delete()
            
        # Clear existing documents in the collection before adding new ones
        docs = db.collection("samples").limit(MAX_DATA_POINTS).stream()
        for doc in docs:
            doc.reference.delete()
        
        # Add new data to Firestore
        batch = db.batch()
        samples_ref = db.collection("samples")
        for data in data_queue:
            batch.set(samples_ref.document(), data)
        
        # Commit the batch operation
        batch.commit()

# Upload the CSV data
upload_csv_to_firestore('./2/Radar_log0.csv')  # Update this path to your CSV file
