import pickle
import shap
import argparse
import pandas as pd
from pathlib import Path

def explain_model(dataset_name, model_name):
    # Load data and model
    with open(f'x_test_{dataset_name}.pickle', 'rb') as f:
        X_test = pickle.load(f)
    with open(f'models/{dataset_name}_{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Create explainer
    if 'Forest' in model_name or 'Tree' in model_name:
        explainer = shap.TreeExplainer(model.named_steps[model.steps[-1][0]])
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_test, 100))
    
    # Generate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Save results
    results_dir = Path(f'results/{dataset_name}/shap')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(results_dir / f'{model_name}_summary.png')
    
    # Save SHAP values
    pd.DataFrame(shap_values, columns=X_test.columns).to_csv(
        results_dir / f'{model_name}_values.csv'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['australian', 'german'], required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    explain_model(args.dataset, args.model)