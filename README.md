# Advanced EDA & Feature Engineering

## Overview

This project focuses on transforming a raw Titanic dataset into a clean and machine learning-ready dataset using Exploratory Data Analysis (EDA), data cleaning, and feature engineering techniques.

The workflow includes handling missing values, detecting and treating outliers, creating new predictive features, and generating visualizations to better understand the dataset. The final cleaned dataset can be used as input for machine learning models.

---

## Features

* Data Loading and Validation
* Missing Value Handling
* Statistical Imputation (Median and Mode)
* Outlier Detection using IQR
* Multiple Outlier Handling Methods (Remove, Clip, Flag)
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Data Visualization
* Clean Dataset Export
* Command Line Interface (CLI) Support

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* argparse
* Logging

---

## Project Structure

```text
Advanced-EDA-Feature-Engineering/
│
├── main.py
├── README.md
├── titanic.txt (Dataset)
├── Cleaned_Titanic.csv
└── plots/
    ├── age_distribution.png
    ├── fare_boxplot.png
    └── correlation_heatmap.png
```

---

## How to Run

1. Clone this repository.

2. Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn
```

3. Place the `titanic.txt` dataset in the project folder.

4. Run the project.

```bash
python main.py
```

---

## Data Processing

The project performs the following preprocessing steps:

* Handles missing values using Median and Mode imputation.
* Detects outliers in the Age column using the IQR method.
* Supports removing, clipping, or flagging outliers.
* Creates new features:
  * FamilySize
  * IsAlone
  * FarePerPerson

---

## Visualizations

The project automatically generates:

* Age Distribution Histogram
* Fare Boxplot
* Correlation Heatmap

These visualizations help in understanding the dataset before model building.

---

## Output

After execution, the project generates:

* Cleaned_Titanic.csv
* Age Distribution Plot
* Fare Boxplot
* Correlation Heatmap

---

## Future Improvements

* Add Mean and KNN Imputation
* Support Z-Score based Outlier Detection
* Automate EDA Report Generation
* Integrate Machine Learning Models

---

## Author

**Kush Sharma**

Machine Learning & Data Science Enthusiast# Advanced EDA & Feature Engineering

## Overview

This project focuses on transforming a raw Titanic dataset into a clean and machine learning-ready dataset using Exploratory Data Analysis (EDA), data cleaning, and feature engineering techniques.

The workflow includes handling missing values, detecting and treating outliers, creating new predictive features, and generating visualizations to better understand the dataset. The final cleaned dataset can be used as input for machine learning models.

---

## Features

* Data Loading and Validation
* Missing Value Handling
* Statistical Imputation (Median and Mode)
* Outlier Detection using IQR
* Multiple Outlier Handling Methods (Remove, Clip, Flag)
* Feature Engineering
* Exploratory Data Analysis (EDA)
* Data Visualization
* Clean Dataset Export
* Command Line Interface (CLI) Support

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* argparse
* Logging

---

## Project Structure

```text
Advanced-EDA-Feature-Engineering/
│
├── main.py
├── README.md
├── titanic.txt (Dataset)
├── Cleaned_Titanic.csv
└── plots/
    ├── age_distribution.png
    ├── fare_boxplot.png
    └── correlation_heatmap.png
```

---

## How to Run

1. Clone this repository.

2. Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn
```

3. Place the `titanic.txt` dataset in the project folder.

4. Run the project.

```bash
python main.py
```

---

## Data Processing

The project performs the following preprocessing steps:

* Handles missing values using Median and Mode imputation.
* Detects outliers in the Age column using the IQR method.
* Supports removing, clipping, or flagging outliers.
* Creates new features:
  * FamilySize
  * IsAlone
  * FarePerPerson

---

## Visualizations

The project automatically generates:

* Age Distribution Histogram
* Fare Boxplot
* Correlation Heatmap

These visualizations help in understanding the dataset before model building.

---

## Output

After execution, the project generates:

* Cleaned_Titanic.csv
* Age Distribution Plot
* Fare Boxplot
* Correlation Heatmap

---

## Future Improvements

* Add Mean and KNN Imputation
* Support Z-Score based Outlier Detection
* Automate EDA Report Generation
* Integrate Machine Learning Models

---

## Author

**Kush Sharma**

