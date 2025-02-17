# A Hybrid Error Correction Model for Prediction of Credit Approval: An XAI Approach

This repository contains the code and data used for the research paper "A Hybrid Error Correction Model for Prediction of Credit Approval: An XAI Approach," published in Elsevier's Engineering Applications of Artificial Intelligence journal. The full paper can be accessed here: [https://doi.org/10.1016/j.engappai.2025.110140](https://doi.org/10.1016/j.engappai.2025.110140).

This research presents a hybrid model integrating error correction mechanisms with eXplainable Artificial Intelligence (XAI) techniques to enhance the prediction accuracy and interpretability of credit approval decisions. The final trained model and relevant datasets are available upon request.

## Repository Structure
* The dataset preprocessing and exploratory data analysis (EDA) scripts are located in scripts/preprocessing.py.
* The main model implementation and training are in scripts/train_models.py.
* Feature selection and explainability techniques are implemented in scripts/xai_feature_selection.py.
## Installation
To set up the environment for running the experiments, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/ElhamDarvish/credit-scoring-analysis.git
   cd hybrid-credit-approval
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Methodology Overview
1. **Data Preprocessing**: Handles missing values, outliers, and normalizes features.
2. **Feature Engineering**: Generates a SHAP-based feature selection approach to identify most impactful features.
3. **Hybrid Model Training**: Combines traditional machine learning techniques with an error correction mechanism.
4. **Evaluation**: Uses metrics like accuracy, precision, recall, F1-score,confusion matrix, and AUC-ROC.
5. **Explainability**: Employs SHAP to interpret model predictions.

## Results and Findings
- The proposed model improves credit approval prediction accuracy compared to baseline models.
- XAI methods provide insight into key decision-making factors in credit approvals.
- Error correction mechanisms help mitigate biases in the dataset.

## License
This code is released under an [MIT License](https://choosealicense.com/licenses/mit/).

## Citation
If you use this code, please cite:
```
@article{darvish_hybrid_2025,
  title = {A Hybrid Error Correction Model for Prediction of Credit Approval: An XAI Approach},
  author = { Elham Darvish, Mustafa Jahangoshai Rezaee, Mohsen Abbaspour Onari},
  journal = {Engineering Applications of Artificial Intelligence},
  year = {2025},
  doi = {(https://doi.org/10.1016/j.engappai.2025.110140)}
}
```

## Contact
For any questions, please open an issue or contact me at [LinkedIn](https://www.linkedin.com/in/elham-darvish).

