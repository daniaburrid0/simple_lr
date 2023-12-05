# Rust Linear Regression Analysis

## Overview
This project is a demonstration of linear regression analysis implemented in Rust. It showcases the use of Rust for data science tasks, particularly focusing on reading, processing, and analyzing data to build and evaluate a linear regression model. The project utilizes the `smartcore` library for machine learning algorithms and `polars` for efficient DataFrame operations.

## Features
- **CSV Data Reading**: Utilizes `polars` to read and process CSV data.
- **Data Preprocessing**: Splits the data into features and target datasets.
- **Matrix Conversion**: Transforms feature data into a DenseMatrix format suitable for machine learning algorithms.
- **Linear Regression Model**: Implements a linear regression model using `smartcore`.
- **Model Evaluation**: Evaluates the model using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Dependencies
- `polars`: For data manipulation and DataFrame operations.
- `smartcore`: Provides machine learning algorithms.
- `std::fs`: For file handling.
- `std::path`: For file path handling.

## Usage
To run this project, ensure you have Rust installed on your system. Clone the repository and navigate to the project directory. Run the project using:

```bash
cargo run
```

Make sure to modify the file path in the `main` function to point to your CSV data file.

## Code Structure
- `read_csv`: Reads a CSV file and returns a DataFrame.
- `feature_and_target`: Splits the DataFrame into separate feature and target DataFrames.
- `convert_features_to_matrix`: Converts the feature DataFrame into a DenseMatrix.
- `main`: Orchestrates the data reading, preprocessing, model training, prediction, and evaluation.

## Data
The dataset used in this project is a CSV file with features and a target variable. Update the file path in the `main` function to point to your dataset.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is open source and available under the [MIT License](LICENSE).