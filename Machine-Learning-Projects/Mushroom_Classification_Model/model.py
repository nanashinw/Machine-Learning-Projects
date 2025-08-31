import pandas as pd
from sklearn.model_selection import train_test_split
RANDOM_STATE = 55
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv("drug200.csv")

cat_variables = ['Sex',
                 'BP',
                 'Cholesterol',
                 'Drug'
]

df = pd.get_dummies(data = df, prefix = cat_variables, columns = cat_variables)
features = [x for x in df.columns if not x.startswith('Drug_drug')]

if 'Drug_drug' not in df.columns:
    df['Drug_drug'] = df[['Drug_drugA', 'Drug_drugB', 'Drug_drugC', 'Drug_drugX', 'Drug_drugY']].idxmax(axis=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Drug_label'] = le.fit_transform(df['Drug_drug'])


X_train, X_val, y_train, y_val = train_test_split(
    df[features], 
    df['Drug_label'], 
    train_size=0.8, 
    random_state=RANDOM_STATE
)

#Fit the model
random_forest_model = RandomForestClassifier(n_estimators = 15,
                                             max_depth = 4, 
                                             min_samples_split = 6).fit(X_train,y_train)


# Make pickle file of our model
pickle.dump(random_forest_model, open("model.pkl", "wb"))

