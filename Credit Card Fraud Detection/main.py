import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset (use a smaller sample if necessary)
df = pd.read_csv(r'C:\Users\jaypa\Projects\Codsoft datasets\Credit Card Fraud Detection\creditcard.csv')
df = df.sample(frac=0.1, random_state=42)  # 10% sample to reduce memory usage

# Preprocessing
X = df.drop(['Class'], axis=1)
y = df['Class']

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



