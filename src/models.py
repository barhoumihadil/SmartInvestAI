import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def check_target_balance(y):
    """
    Affiche la distribution des classes et retourne les proportions
    """
    counts = y.value_counts()
    props = y.value_counts(normalize=True)
    print("=== Distribution de la target ===")
    print(pd.concat([counts, props], axis=1, keys=['Count','Proportion']))
   
    plt.figure(figsize=(6,4))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title("Répartition des classes de Target")
    plt.ylabel("Nombre d'observations")
    plt.show()
    
    return counts, props

def train_models_grid(X, y, test_size=0.2, random_state=42):
    """
    Entraîne 5 modèles supervisés avec GridSearchCV et retourne :
    - best_models : dictionnaire des meilleurs modèles
    - results : dictionnaire des métriques et meilleurs paramètres
    """
   
    check_target_balance(y)
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    
    models_params = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=random_state, class_weight='balanced'),
            "params": {'n_estimators':[50,100], 'max_depth':[None,5], 'min_samples_split':[2,5]}
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
            "params": {'n_estimators':[50,100], 'max_depth':[3,5], 'learning_rate':[0.01,0.1]}
        },
        "GradientBoost": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "params": {'n_estimators':[50,100], 'learning_rate':[0.05,0.1], 'max_depth':[3,5]}
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=500, random_state=random_state, class_weight='balanced'),
            "params": {'C':[0.1,1,10], 'solver':['lbfgs']}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=random_state, class_weight='balanced'),
            "params": {'C':[0.1,1,10], 'kernel':['linear','rbf']}
        }
    }

    results = {}
    best_models = {}

    for name, mp in models_params.items():
        print(f"Entraînement et Grid Search pour {name}...")
        grid = GridSearchCV(mp["model"], mp["params"], cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:,1]
        
        results[name] = {
            "best_params": grid.best_params_,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        best_models[name] = best_model
        print(f"{name} terminé.\nAccuracy: {results[name]['accuracy']:.4f} | F1: {results[name]['f1']:.4f} | ROC-AUC: {results[name]['roc_auc']:.4f}")
    
    results_df = pd.DataFrame(results).T
    print("\n=== Résultats finaux ===")
    print(results_df)
    
    return best_models, results
