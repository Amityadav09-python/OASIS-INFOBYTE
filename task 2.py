import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:\Users\Amit\Downloads\Unemployment in India.csv")
print(data)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display basic information about the dataset
print("\nBasic information about the dataset:")
print(data.info())

# Display summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Data Cleaning (if necessary)
# For example, filling missing values with the mean of the column
data = data.fillna(data.mean())

# Data Analysis
# Plotting the unemployment rate over time
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Unemployment Rate'], marker='o')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Correlation analysis
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Additional analysis can be performed as needed
# For example, distribution of unemployment rates
plt.figure(figsize=(10, 6))
sns.histplot(data['Unemployment Rate'], kde=True)
plt.title('Distribution of Unemployment Rates')
plt.xlabel('Unemployment Rate')
plt.ylabel('Frequency')
plt.show()

# If the dataset contains categorical data, you can analyze those as well
# For example, unemployment rate by region (if 'Region' is a column in the dataset)
if 'Region' in data.columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Region', y='Unemployment Rate', data=data)
    plt.title('Unemployment Rate by Region')
    plt.xlabel('Region')
    plt.ylabel('Unemployment Rate')
    plt.xticks(rotation=45)
    plt.show()