# Smartphone Price Tier Classification

This project uses machine learning to classify smartphones into different price tiers (Budget, Mid-Range, and Flagship) based on their technical specifications. The analysis involves data cleaning, exploratory data analysis (EDA), feature engineering, and a comparative evaluation of several classification models. The **Random Forest Classifier** was identified as the best-performing model, achieving an average accuracy of **86.6%** in cross-validation.

## 📂 Table of Contents
* [Project Goal](#-project-goal)
* [Dataset](#-dataset)
* [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
* [Methodology](#-methodology)
* [Model Performance](#-model-performance)
* [Key Findings](#-key-findings)
* [Technologies Used](#️-technologies-used)
* [How to Run This Project](#-how-to-run-this-project)

## 🎯 Project Goal
The primary goal of this project is to build and evaluate a machine learning model that can accurately predict a smartphone's price category (Budget, Mid-Range, or Flagship) using its features, such as brand, processor, camera quality, and battery capacity.

## 🧾 Dataset
The project utilizes the `Smartphones_cleaned_dataset.csv` file. This dataset contains various technical specifications for numerous smartphone models.

**Key Features in the Dataset:**
- `brand_name`
- `price` and `rating`
- `has_5g`, `has_nfc`, `has_ir_blaster`
- `processor_brand`, `num_cores`, `processor_speed`
- `battery_capacity`, `ram_capacity`, `internal_memory`
- `primary_camera_rear`, `primary_camera_front`

The initial data required cleaning to handle missing values, which were imputed using median for numerical columns and mode for categorical columns.

## 📊 Exploratory Data Analysis (EDA)
Several visualizations were created to understand the relationships and distributions within the data:
1.  **Price Distribution**: A histogram showed the frequency distribution of smartphone prices.
2.  **Price vs. Rating**: A scatter plot was used to examine the correlation between price and user ratings.
3.  **Top Brands**: A bar chart highlighted the top 10 brands with the highest number of models in the dataset.
4.  **Feature Correlation**: A heatmap visualized the correlation between the different numerical features of the smartphones.

![EDA Plots](SoM_Project.ipynb_files/SoM_Project_12_1.png)
*Correlation Heatmap and Top Brands Plot*

## ⚙️ Methodology
The project follows a standard machine learning workflow:
1.  **Data Cleaning**: Missing values were handled by imputing with median for numerical features and mode for categorical ones.
2.  **Feature Engineering**: A target variable, `price_tier`, was created by categorizing the `price` column into three classes:
    - **Budget**: price < 20,000
    - **Mid-Range**: 20,000 <= price < 50,000
    - **Flagship**: price >= 50,000
3.  **Preprocessing**: A `ColumnTransformer` pipeline was built to scale numerical features using `StandardScaler` and encode categorical features using `OneHotEncoder`.
4.  **Model Comparison**: Five different classification models were trained and evaluated using 5-fold cross-validation:
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Random Forest
    - Naive Bayes
    - Support Vector Machine (SVM)

## 🏆 Model Performance
The cross-validation results showed that the **Random Forest** model performed the best.

| Model               | Avg. Accuracy | Std Deviation |
|---------------------|---------------|---------------|
| **Random Forest** | **0.8663** | **0.0234** |
| SVM                 | 0.8582        | 0.0231        |
| K-Nearest Neighbors | 0.8469        | 0.0326        |
| Decision Tree       | 0.8367        | 0.0141        |
| Naive Bayes         | NaN           | NaN           |

### Detailed Analysis of Random Forest
The best model (Random Forest) was further analyzed on a held-out test set.
- **Confusion Matrix**: This shows the model's performance for each class, highlighting where it makes correct and incorrect predictions.
- **ROC Curve**: The multi-class ROC curve illustrates the model's ability to distinguish between the different price tiers.

![Confusion Matrix](SoM_Project.ipynb_files/SoM_Project_18_1.png)
![ROC Curve](SoM_Project.ipynb_files/SoM_Project_18_2.png)

## 🔑 Key Findings
- The technical specifications of a smartphone are strong predictors of its price category.
- The **Random Forest** algorithm is the most effective for this classification task among the models tested, achieving a high accuracy of **86.6%**.
- The model performs well in distinguishing between the 'Budget', 'Mid-Range', and 'Flagship' classes, as shown by the confusion matrix and high AUC scores in the ROC analysis.

## 🛠️ Technologies Used
- **Python**
- **Pandas** for data manipulation and analysis.
- **NumPy** for numerical operations.
- **Matplotlib** & **Seaborn** for data visualization.
- **Scikit-learn** for machine learning, including preprocessing, model training, and evaluation.
- **Jupyter Notebook** for interactive development.

## 🚀 How to Run This Project
To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Suchendra13/SoM_Project.git](https://github.com/Suchendra13/SoM_Project.git)
    cd SoM_Project
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
4.  Open the `SoM_Project.ipynb` file and run the cells sequentially. Ensure the `Smartphones_cleaned_dataset.csv` file is in the same directory.
