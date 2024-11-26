import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv(r'C:\Users\jaypa\Projects\Codsoft datasets\Sales Prediction\advertising.csv')

# Preprocessing
X = df[['TV', 'Radio', 'Newspaper']]
X = pd.get_dummies(X, drop_first=True)  # Encoding categorical variables
y = df['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
