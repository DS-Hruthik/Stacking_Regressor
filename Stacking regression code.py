# Stacking Regression Using scikit-learn
from sklearn.datasets import load_diabetes  # Importing dataset
from sklearn.linear_model import RidgeCV  # Ridge regression
from sklearn.svm import LinearSVR  # Support Vector Regression
from sklearn.ensemble import RandomForestRegressor  # Random Forest Regression
from sklearn.ensemble import StackingRegressor  # Stacking Regressor
from sklearn.model_selection import train_test_split  # Train test split
import pandas as pd
import numpy as np
import pickle  # For saving and loading models

# Load the dataset
diabetes = load_diabetes()

df_features = pd.DataFrame(data = diabetes.data, columns = diabetes.feature_names)
df_target = pd.DataFrame(data = diabetes.target, columns = ['target'])

# base data set in built 
final = pd.concat([df_features, df_target], axis = 1)
final

# dividing by columns 
X = np.array(final.iloc[:,:10]) #predictors
y = np.array(final['target'])

# dividing by rows 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42)


#models: these are base
estimators = [("lr", RidgeCV()), ("svr", LinearSVR(random_state=42))]

# metal learner:(as a random forest)
reg = StackingRegressor(estimators = estimators,
                        final_estimator = RandomForestRegressor(n_estimators = 10,
                                                                random_state = 42))

stacking_reg= reg.fit(X_train, y_train)

pickle.dump(stacking_reg, open('stacking_reg_diabetes.pkl', 'wb'))

#loading above pickle file 
model = pickle.load(open('stacking_reg_diabetes.pkl', 'rb'))

pred_test = model.predict(X_test)
pred_test

r2_score = model.score(X_test, y_test) # something you will learn in next topic regression
r2_score

# using some other new data to predict
test = pd.read_csv(r"C:\Users\hruth\Desktop\self pace learning\course\ML\j Ensemble Technique\Voting and Stacking Material\4.d.Ensemble Models\Voting\breast_cancer_test.csv")
test_pred = model.predict(test)















































































