import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score

# Set style for plots - using a valid style
plt.style.use('ggplot')  # Changed from 'seaborn' to 'ggplot'
sns.set_palette("husl")

# Load the dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    data = pd.read_csv(url, header=None, names=column_names, na_values=' ?', skipinitialspace=True)
    
    # Clean column names
    data.columns = [col.replace('-', '_') for col in data.columns]
    
    # Convert target variable to binary
    data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    return data

# Preprocessing
def preprocess_data(data):
    # Define features and target
    X = data.drop('income', axis=1)
    y = data['income']
    
    # Define numerical and categorical features
    numerical_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 
                           'relationship', 'race', 'sex', 'native_country']
    
    # Preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Fix for OneHotEncoder compatibility
    try:
        # For scikit-learn >= 1.2
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except TypeError:
        # For older scikit-learn versions
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

# Train and evaluate models
def train_and_evaluate_models(X, y, preprocessor):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Classifier': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
        
        # Store results
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"{name} trained successfully!")
    
    return results, X_test, y_test

# Generate evaluation plots
def generate_evaluation_plots(results, X_test, y_test):
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Model Comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(results.keys())
    
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = [results[model][metric] for model in model_names]
        plt.bar(model_names, values)
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xticks(rotation=45)
        plt.ylabel(metric.title())
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    # Plot 2: Confusion Matrix for Best Model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    cm = results[best_model_name]['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['≤50K', '>50K'], 
                yticklabels=['≤50K', '>50K'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Plot 3: ROC Curve
    plt.figure(figsize=(8, 6))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('plots/roc_curve.png')
    plt.close()
    
    # Plot 4: Feature Importance (for tree-based models)
    if 'Gradient Boosting' in results or 'Random Forest' in results:
        tree_models = []
        if 'Gradient Boosting' in results:
            tree_models.append(('Gradient Boosting', results['Gradient Boosting']))
        if 'Random Forest' in results:
            tree_models.append(('Random Forest', results['Random Forest']))
        
        for name, result in tree_models:
            model = result['model'].named_steps['classifier']
            preprocessor = result['model'].named_steps['preprocessor']
            
            # Get feature names
            try:
                # For newer scikit-learn versions
                num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
            except AttributeError:
                # For older scikit-learn versions
                num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
            
            feature_names = np.concatenate([num_features, cat_features])
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Sort and plot
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importances - {name}')
            plt.barh(range(len(indices)), importances[indices][::-1], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
            plt.xlabel('Relative Importance')
            plt.tight_layout()
            plt.savefig(f'plots/feature_importance_{name.lower().replace(" ", "_")}.png')
            plt.close()

# Generate classification report
def generate_classification_report(results):
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    y_test = results[best_model_name]['y_test']
    y_pred = results[best_model_name]['y_pred']
    
    report = classification_report(y_test, y_pred, target_names=['≤50K', '>50K'])
    
    with open('classification_report.txt', 'w') as f:
        f.write(f"Classification Report - {best_model_name}\n")
        f.write("="*50 + "\n\n")
        f.write(report)
        f.write(f"\nAccuracy: {results[best_model_name]['accuracy']:.4f}\n")
        f.write(f"Precision: {results[best_model_name]['precision']:.4f}\n")
        f.write(f"Recall: {results[best_model_name]['recall']:.4f}\n")
        f.write(f"F1-Score: {results[best_model_name]['f1']:.4f}\n")
        f.write(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}\n")
        f.write(f"Cross-Validation Mean Accuracy: {results[best_model_name]['cv_mean']:.4f} (±{results[best_model_name]['cv_std']:.4f})\n")

# Save the best model
def save_best_model(results):
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    joblib.dump(best_model, 'model.joblib')
    print(f"Best model ({best_model_name}) saved as 'model.joblib'")

# Main function
def main():
    print("Starting Employee Salary Prediction Model Training...")
    
    # Load data
    print("Loading data...")
    data = load_data()
    print(f"Dataset shape: {data.shape}")
    print(f"Class distribution:\n{data['income'].value_counts(normalize=True)}")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, preprocessor = preprocess_data(data)
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results, X_test, y_test = train_and_evaluate_models(X, y, preprocessor)
    
    # Generate evaluation plots
    print("Generating evaluation plots...")
    generate_evaluation_plots(results, X_test, y_test)
    
    # Generate classification report
    print("Generating classification report...")
    generate_classification_report(results)
    
    # Save best model
    print("Saving best model...")
    save_best_model(results)
    
    # Print summary
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    print("\nModel Performance Summary:")
    print("-"*50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1-Score: {result['f1']:.4f}")
        print(f"  ROC AUC: {result['roc_auc']:.4f}")
        print(f"  CV Accuracy: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
    
    print(f"\nBest Model: {max(results.keys(), key=lambda x: results[x]['accuracy'])}")
    print(f"Plots saved in 'plots/' directory")
    print(f"Classification report saved as 'classification_report.txt'")

if __name__ == "__main__":
    main()