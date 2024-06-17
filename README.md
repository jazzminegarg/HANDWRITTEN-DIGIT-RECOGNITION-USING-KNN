# HANDWRITTEN-DIGIT-RECOGNITION-USING-KNN
Train your model using K-Nearest Neighbor Algorithm with having values of K as {2,4,5,6,7,10}, over data.csv file provided. The Train and Test split of the data should be in the ratio of 60:40, 70:30, 75:25, 80:20, 90:10, 95:5. Evaluate the performance of the model over test data for all these scenarios (36 cases)
Here’s a README for your GitHub repository:

markdown
Copy code
# K-Nearest Neighbor Model Evaluation

## Overview

This repository contains a study on the performance of the K-Nearest Neighbor (KNN) algorithm with different train-test splits and K values on a provided dataset. The study evaluates 36 scenarios and analyzes the dependency of model performance on training-testing split and K value.

## Files

- `knn_evaluation.ipynb`: Jupyter Notebook with the implementation of one scenario and comments for others.
- `knn_results.csv`: CSV file containing the results (accuracy and confusion matrix) for all scenarios.
- `knn_results.pdf`: PDF report with the results and analysis of the study.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/knn-evaluation.git
Navigate to the directory:
bash
Copy code
cd knn-evaluation
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook:
Open knn_evaluation.ipynb in Jupyter Notebook and run the cells to see the example implementation.
Results
The results of the study, including accuracy and confusion matrices for all scenarios, are documented in the knn_results.pdf file. The analysis shows the impact of different train-test splits and K values on the performance of the KNN model.

License
This project is licensed under the MIT License.

markdown
Copy code

### Additional Steps

- **Create a `requirements.txt` file** for the dependencies:
  ```bash
  pip freeze > requirements.txt
Generate the PDF report using a tool like LaTeX or any PDF editor by compiling the results and analysis.
