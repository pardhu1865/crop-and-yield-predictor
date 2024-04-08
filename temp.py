import pandas as pd

# Read the dataset from a CSV file
data = pd.read_csv('crop_production.csv')

# Define the desired crops
desired_crops = ["rice", "maize", "banana", "coconut", "grapes", "mango", "apple", "orange", "papaya", "pomegranate", "jute", "coffee"]

# Filter the dataset to include only rows with the specified crops
filtered_data = data[data['Crop'].str.lower().isin(desired_crops)]

# Get unique district names from the filtered dataset
districts_with_desired_crops = filtered_data['District_Name'].unique()

# Generate HTML <option> tags with line breaks
html_options = '\n'.join([f'<option value="{district}">{district}</option>' for district in districts_with_desired_crops])

# Print or use the HTML options
print(html_options)
