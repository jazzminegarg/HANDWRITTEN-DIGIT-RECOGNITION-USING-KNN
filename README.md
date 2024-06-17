Certainly! Below is a README file for the provided code, suitable for a GitHub repository.

---

# K-Nearest Neighbor Model Evaluation

This repository contains a study on the performance of the K-Nearest Neighbor (KNN) algorithm with different train-test splits and K values on a provided dataset. The study evaluates various scenarios and analyzes the dependency of model performance on training-testing split and K value.

## Overview

The provided code trains a K-Nearest Neighbor (KNN) model on the `data.csv` dataset. The model is evaluated with different values of K and various train-test splits. The results, including accuracy and confusion matrices, are saved to a PDF file.

## Files

- `hand_digit_recognition.ipynb`: Jupyter Notebook with the implementation of one scenario and comments for others.
- `data.csv`: The dataset used for training and evaluation.
- `knn_results.pdf`: PDF report with the results (accuracy and confusion matrix) for all scenarios.
- `knn_results1.pdf`: PDF report with the results for one example scenario.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- ReportLab

You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib reportlab
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jazzmine_garg/HANDWRITTEN_DIGIT_RECOGNITION_USING_KNN.git
   ```
3. **Place your `data.csv` file in the directory**.

4. **Run the script**:

   - **For a single scenario**:
     The code provided in the script evaluates one scenario where the train-test split is 70:30 and K is 2.
     ```python
     python knn_evaluation.py
     ```
   - **For all scenarios**:
     Uncomment the full implementation section in the script to evaluate all scenarios and save results to `knn_results.pdf`.

5. **View the results**:
   Open `knn_results.pdf` to see the accuracy and confusion matrices for all scenarios. For the single example scenario, check `knn_results1.pdf`.

## Code Explanation

### Data Preparation

The code starts by loading the dataset from `data.csv` and splitting it into features (`X`) and target (`y`).

```python
data = pd.read_csv('data.csv')
X = data.drop(columns='label')
y = data['label']
```

### Model Evaluation Function

A helper function `evaluate_knn` is defined to train and evaluate the KNN model.

```python
def evaluate_knn(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix
```

### Example Scenario

An example scenario is provided where the train-test split is 70:30 and K is 2.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
accuracy, conf_matrix = evaluate_knn(X_train, X_test, y_train, y_test, 2)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
```

### Full Implementation

The full implementation is commented out to avoid long runtime. Uncomment these sections to evaluate all scenarios.

```python
# for split in train_test_splits:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
#     for k in k_values:
#         accuracy, conf_matrix = evaluate_knn(X_train, X_test, y_train, y_test, k)
#         results[(split, k)] = (accuracy, conf_matrix)
```

### Saving Results

The results for the example scenario are saved to a PDF file.

```python
pdf = matplotlib.backends.backend_pdf.PdfPages("knn_results1.pdf")
fig, ax = plt.subplots()
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title(f"Train-Test Split: {0.3}, K: {2}\nAccuracy: {accuracy:.2f}")
pdf.savefig(fig)
plt.close()
pdf.close()
```

## Analysis

The analysis section in the PDF report discusses the impact of different train-test splits and K values on the model's performance. It helps in understanding how these parameters affect accuracy and model behavior.

## License

This project is licensed under the MIT License.
