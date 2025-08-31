import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib

RANDOM_STATE = 55

# Load and preprocess data
print("Loading and preprocessing data...")
df_train = pd.read_csv("train.csv")

# Handle missing values
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

# Encode categorical variables
cat_cols = df_train.select_dtypes(include='object').columns.tolist()
label_encoders = {}

# Remove columns we won't use but preserve Sex for encoding
columns_to_remove = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked']
for col in columns_to_remove:
    if col in df_train.columns:
        df_train.drop(col, axis=1, inplace=True)

# Encode Sex column (this is the main categorical we need)
if 'Sex' in df_train.columns:
    le_sex = LabelEncoder()
    df_train['Sex'] = le_sex.fit_transform(df_train['Sex'])
    label_encoders['Sex'] = le_sex

# Prepare features and target
y = df_train["Survived"].to_numpy()
X = df_train.drop('Survived', axis=1).to_numpy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {df_train.drop('Survived', axis=1).columns.tolist()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Train Neural Network Model
print("\nTraining Neural Network Model...")
tf.random.set_seed(RANDOM_STATE)

# Fixed model architecture for binary classification
model_nn = Sequential([
    Dense(120, activation='relu', input_shape=(X_train.shape[1],), 
          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
], name="TitanicSurvivalModel")

model_nn.compile(
    loss='binary_crossentropy',  # Better for binary classification
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train the model
history = model_nn.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate Neural Network
nn_predictions = model_nn.predict(X_test)
nn_predictions_binary = (nn_predictions > 0.5).astype(int).flatten()
nn_accuracy = accuracy_score(y_test, nn_predictions_binary)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

# Train Random Forest for comparison
print("\nTraining Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=10)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Choose the best model
if nn_accuracy >= rf_accuracy:
    best_model = model_nn
    best_model_name = "Neural Network"
    best_accuracy = nn_accuracy
    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
else:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_accuracy
    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Save the best model
print(f"Saving {best_model_name} model...")

if best_model_name == "Neural Network":
    # Save TensorFlow model
    best_model.save("titanic_model.h5")
    # Also save as pickle for consistency with Flask app
    pickle.dump(best_model, open("model.pkl", "wb"))
else:
    # Save scikit-learn model
    joblib.dump(best_model, "model.pkl")

# Save label encoders
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

# Print classification report
if best_model_name == "Neural Network":
    print("\nClassification Report (Neural Network):")
    print(classification_report(y_test, nn_predictions_binary))
else:
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, rf_predictions))

# Feature importance for Random Forest
if best_model_name == "Random Forest":
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

print(f"\nModel saved successfully! You can now run the Flask app.")
print("Make sure to place the HTML template in templates/index.html")

# Test prediction function
print("\nTesting prediction with sample data...")
sample_data = np.array([[3, 1, 25.0, 1, 0]])  # 3rd class, male, 25 years, 1 sibling, 0 parents
if best_model_name == "Neural Network":
    test_prediction = best_model.predict(sample_data)[0][0]
    test_result = "Survived" if test_prediction > 0.5 else "Did Not Survive"
    print(f"Sample prediction: {test_result} (confidence: {test_prediction:.3f})")
else:
    test_prediction = best_model.predict(sample_data)[0]
    test_result = "Survived" if test_prediction == 1 else "Did Not Survive"
    print(f"Sample prediction: {test_result}")