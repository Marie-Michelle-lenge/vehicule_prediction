import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

_df = pd.read_csv('data/dataset_DGI_ready.csv')
excluded_cols = [
    'assurance', 'controle_technique', 'vignette',
    'taxe_circulation_usd', 'taxe_stationnement_usd', 'age_vehicule',
    'charge_fiscale', 'score', 'score_final', 'statut_fiscal'
]
X = _df.drop(columns=['statut_fiscal_encoded'] + excluded_cols)
y = _df['statut_fiscal_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
num_preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
preprocessor_lr = ColumnTransformer([('cat', cat_preprocessor, cat_cols), ('num', num_preprocessor, num_cols)])
preprocessor_tree = ColumnTransformer([('cat', cat_preprocessor, cat_cols), ('num', SimpleImputer(strategy='median'), num_cols)])

models = []
model_lr = Pipeline([('preprocessor', preprocessor_lr), ('model', LogisticRegression(max_iter=2000, random_state=42))])
search_lr = GridSearchCV(model_lr, {'model__C':[0.1,1.0,10.0], 'model__solver':['lbfgs'], 'model__penalty':['l2']}, cv=5, scoring='f1_weighted', n_jobs=1)
search_lr.fit(X_train, y_train)
models.append(('Logistic Regression', search_lr.best_estimator_, search_lr.best_params_, cross_val_score(search_lr.best_estimator_, X_train, y_train, cv=5, scoring='f1_weighted').mean()))

model_rf = Pipeline([('preprocessor', preprocessor_tree), ('model', RandomForestClassifier(random_state=42))])
search_rf = GridSearchCV(model_rf, {'model__n_estimators':[100, 200], 'model__max_depth':[5, 10, None], 'model__min_samples_split':[2,5], 'model__min_samples_leaf':[1,2]}, cv=5, scoring='f1_weighted', n_jobs=1)
search_rf.fit(X_train, y_train)
models.append(('Random Forest', search_rf.best_estimator_, search_rf.best_params_, cross_val_score(search_rf.best_estimator_, X_train, y_train, cv=5, scoring='f1_weighted').mean()))

model_dt = Pipeline([('preprocessor', preprocessor_tree), ('model', DecisionTreeClassifier(random_state=42))])
search_dt = GridSearchCV(model_dt, {'model__max_depth':[3,5,None], 'model__min_samples_split':[2,5,10], 'model__min_samples_leaf':[1,2], 'model__ccp_alpha':[0.0,0.01]}, cv=5, scoring='f1_weighted', n_jobs=1)
search_dt.fit(X_train, y_train)
models.append(('Decision Tree', search_dt.best_estimator_, search_dt.best_params_, cross_val_score(search_dt.best_estimator_, X_train, y_train, cv=5, scoring='f1_weighted').mean()))

for name, model, params, cv_mean in models:
    pred = model.predict(X_test)
    print(name)
    print('  best_params=', params)
    print('  cv_f1_weighted_mean=', round(cv_mean, 6))
    print('  test_accuracy=', round(accuracy_score(y_test, pred), 6))
    print('  test_f1_weighted=', round(f1_score(y_test, pred, average='weighted'), 6))
