import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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

model_lr = Pipeline([('preprocessor', preprocessor_lr), ('model', LogisticRegression(max_iter=2000, random_state=42))])
model_rf = Pipeline([('preprocessor', preprocessor_tree), ('model', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None, min_samples_split=2))])
model_dt = Pipeline([('preprocessor', preprocessor_tree), ('model', DecisionTreeClassifier(random_state=42, max_depth=None, min_samples_split=2))])

for name, model in [('Logistic Regression', model_lr), ('Random Forest', model_rf), ('Decision Tree', model_dt)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, 'accuracy=', round(accuracy_score(y_test, y_pred), 6), 'f1_weighted=', round(f1_score(y_test, y_pred, average='weighted'), 6), 'balanced=', round(balanced_accuracy_score(y_test, y_pred), 6))
