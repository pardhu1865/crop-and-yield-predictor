import pandas as pd

# Read the dataset
crop_data = pd.read_csv('crop_production.csv')

# Extract unique district names and sort them alphabetically
district_names = sorted(crop_data['District_Name'].unique())

# Generate HTML options
html_options = "\n".join([f'<option value="{name}">{name}</option>' for name in district_names])

# Write HTML options to a file
with open('district_options.html', 'w') as file:
    file.write(html_options)
