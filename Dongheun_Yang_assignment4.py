# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump


data_dongheun = pd.read_csv(r'C:\Users\danie\Desktop\Term4\AI\Assignment\Lab Assignment4_Decision Trees\student-por.csv', sep=';')

column_info = data_dongheun.dtypes
missing_values = data_dongheun.isnull().sum()
numeric_stats = data_dongheun.describe()

categorical_columns = data_dongheun.select_dtypes(include=['object']).columns
categorical_values = {col: data_dongheun[col].unique() for col in categorical_columns}

data_dongheun['pass_dongheun'] = ((data_dongheun['G1'] + data_dongheun['G2'] + data_dongheun['G3']) >= 35).astype(int)

data_dongheun.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)

features_dongheun = data_dongheun.drop('pass_dongheun', axis=1)
target_variable_dongheun = data_dongheun['pass_dongheun']

class_distribution = data_dongheun['pass_dongheun'].value_counts()
print(f'class_distribution : {class_distribution}')

numeric_features_dongheun = data_dongheun.select_dtypes(include=['int64', 'float64']).columns.tolist()

cat_features_dongheun = data_dongheun.select_dtypes(include=['object']).columns.tolist()



transformer_dongheun = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(), cat_features_dongheun) 
    ],
    remainder='passthrough'
)

clf_dongheun = DecisionTreeClassifier(criterion="entropy", max_depth=5)

pipeline_dongheun = Pipeline([
    ('transformer', transformer_dongheun),  
    ('classifier', clf_dongheun)            
])



X_train_dongheun, X_test_dongheun, y_train_dongheun, y_test_dongheun = train_test_split(
    features_dongheun,
    target_variable_dongheun,    
    test_size=0.2,
    random_state = 34
)

pipeline_dongheun.fit(X_train_dongheun, y_train_dongheun)
transformed_features = pipeline_dongheun.named_steps['transformer'].get_feature_names_out()


kf = KFold(n_splits=10, shuffle=True, random_state=34)
cv_scores = cross_val_score(pipeline_dongheun, X_train_dongheun, y_train_dongheun, cv=kf)

print("Cross-validation scores for each fold:", cv_scores)

print("Mean cross-validation score: {:.2f}".format(np.mean(cv_scores)))
print("95% confidence interval of the score estimate: +/- {:.2f}".format(np.std(cv_scores) * 2))

decision_tree_model = pipeline_dongheun.named_steps['classifier']
dot_data = export_graphviz(decision_tree_model, out_file=None,
                           feature_names=transformed_features,
                           class_names=['Fail', 'Pass'],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree_visualization")

y_train_pred = pipeline_dongheun.predict(X_train_dongheun)
y_test_pred = pipeline_dongheun.predict(X_test_dongheun)

train_accuracy = accuracy_score(y_train_dongheun, y_train_pred)
print(f"Train Set Accuracy: {train_accuracy:.4f}")

test_accuracy = accuracy_score(y_test_dongheun, y_test_pred)
print(f"Testing Set Accuracy: {test_accuracy:.4f}")

accuracy = accuracy_score(y_test_dongheun, y_test_pred)
precision = precision_score(y_test_dongheun, y_test_pred, average='binary')
recall = recall_score(y_test_dongheun, y_test_pred, average='binary')

conf_matrix = confusion_matrix(y_test_dongheun, y_test_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

parameters = {
    'classifier__min_samples_split': range(10, 300, 20),
    'classifier__max_depth': range(1, 30, 2),
    'classifier__min_samples_leaf': range(1, 15, 3)
}

random_search = RandomizedSearchCV(
    estimator=pipeline_dongheun,
    param_distributions=parameters,
    n_iter=7,  
    scoring='accuracy',
    cv=5,
    refit=True,
    verbose=3,
    random_state=42 
)


random_search.fit(X_train_dongheun, y_train_dongheun)

print("Best parameters found:", random_search.best_params_)

best_model = random_search.best_estimator_

y_test_pred = best_model.predict(X_test_dongheun)
test_accuracy = accuracy_score(y_test_dongheun, y_test_pred)
print(f"Test Accuracy with Best Parameters: {test_accuracy:.4f}")

random_search.fit(X_train_dongheun, y_train_dongheun)

best_parameters = random_search.best_params_
print("Best parameters found:", best_parameters)

best_score_random_search = random_search.best_score_
print(f"Best score from RandomizedSearchCV: {best_score_random_search:.4f}")
mean_cv_score = np.mean(cv_scores)
print(f"Mean cross-validation score before tuning: {mean_cv_score:.4f}")

best_estimator = random_search.best_estimator_
print("Best Estimator:", best_estimator)

y_test_pred = best_estimator.predict(X_test_dongheun)



accuracy = accuracy_score(y_test_dongheun, y_test_pred)
precision = precision_score(y_test_dongheun, y_test_pred, average='binary')
recall = recall_score(y_test_dongheun, y_test_pred, average='binary')
conf_matrix = confusion_matrix(y_test_dongheun, y_test_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

precision = precision_score(y_test_dongheun, y_test_pred, average='binary')
recall = recall_score(y_test_dongheun, y_test_pred, average='binary')
accuracy = accuracy_score(y_test_dongheun, y_test_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

model_filename = 'dongheun_best_model.pkl'
dump(best_estimator, model_filename)

print(f"Model saved to {model_filename}")

pipeline_filename = 'dongheun_pipeline.pkl'
dump(pipeline_dongheun, pipeline_filename)

print(f"Pipeline saved to {pipeline_filename}")