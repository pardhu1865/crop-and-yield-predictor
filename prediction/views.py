from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
from sklearn.metrics import r2_score

# Create your views here.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file
crop_data = pd.read_csv('crop.csv')

desired_crops = ["rice", "maize", "banana", "coconut", "grapes", "mango", "apple", "orange", "papaya", "pomegranate", "jute", "coffee"]

# Filter the dataset to include only rows with the specified crops
filtered_data = crop_data[crop_data['label'].str.lower().isin(desired_crops)]


# Separate features (X) and target labels (y)
X = filtered_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y =filtered_data['label']  # Target labels

# Split the crop_data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree classifier on the training crop_data
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the testing crop_data
dt_predictions = dt_classifier.predict(X_test)

# Evaluate the model's performance on the testing crop_data
# report = classification_report(y_test, dt_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
# print("Classification Report for Decision Tree:")
# print(report)


# def get_production(pro_data):
     

def get_predictions(new_data_point):
    # Use the provided new_data_point for predictions
    new_data = pd.DataFrame([new_data_point])
    dt_predictions = dt_classifier.predict(new_data)

    return {'predictions': dt_predictions.tolist()}

def get_production(prod_data_point):
    # Load the dataset from a CSV file
    total_data = pd.read_csv('crop_production.csv')
    desired_crops = ["Rice", "Maize", "Banana", "Coconut", "Grapes", "Mango", "Apple", "Orange", "Papaya", "Pomegranate", "Jute", "Coffee"]
    print("Accuracy:", dt_accuracy)

# Filter the dataset to include only rows with the specified crops
    data = total_data[total_data['Crop'].isin(desired_crops)]

# Print the size of the filtered data
    # Handle missing values in the dataset
    data.dropna(inplace=True)
    # Separate features (X) and target labels (y)
    X = data[['District_Name', 'Season', 'Crop', 'Area']]
    y = data['Production']

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)
    # print(X_encoded)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Create and train the Decision Tree regressor on the training data
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train, y_train)

    # Convert the new data point to a DataFrame
    new_data = pd.DataFrame([prod_data_point])
  
    # One-hot encode the new data point
    new_data_encoded = pd.get_dummies(new_data)

    # Align the columns to ensure consistency between training and test data
    new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Make predictions for the new data point
    production_prediction = dt_regressor.predict(new_data_encoded)

# Make predictions for the test set
    y_pred = dt_regressor.predict(X_test)

    # Calculate R-squared score (accuracy)
    accuracy = r2_score(y_test, y_pred)

    # Print the accuracy
    print("R-squared score (accuracy):", accuracy)

    return {'production':production_prediction}

    # return {'production': production_prediction}
def get_csrf_token(request):
    # Get the CSRF token from the cookie
    # Return the token in the response
    return JsonResponse({'csrf_token': csrf_token})

def home(request):
    return render(request, 'index.html')

def index(request):
    return render(request, 'form.html')
def predict(request):
    
    # Extract crop_data from the POST request
        new_data_point = {
                'N': float(request.POST.get('nit')),
                'P': float(request.POST.get('phos')),
                'K': float(request.POST.get('pott')),
                'temperature': float(request.POST.get('temp')),
                'humidity': float(request.POST.get('hum')),
                'ph': float(request.POST.get('ph')),
                'rainfall': float(request.POST.get('fall'))
            }


        # Perform predictions using the get_predictions function (update as needed)
        predictions_info = get_predictions(new_data_point)
        crop = predictions_info['predictions'][0]
        capitalized_crop = crop.capitalize()
        print(capitalized_crop)
        predictions_info['predictions'][0]=capitalized_crop
        prod_data_point = {
                'District_Name': request.POST.get('district'),
                'Season': request.POST.get('season'),
                'Crop': capitalized_crop,
                'Area': float(request.POST.get('area'))
            }
        prod_predictions_info = get_production(prod_data_point)
        predictions_info['production'] = prod_predictions_info['production'][0]

        print(predictions_info)
        # Render the predict.html template with the prediction information
        return JsonResponse({'predictions_info': predictions_info})