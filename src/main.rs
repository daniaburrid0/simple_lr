use polars::error::PolarsError;
use polars::prelude::*;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::metrics::{mean_absolute_error, mean_squared_error};
use smartcore::model_selection::train_test_split;
use std::fs::File;
use std::path::Path;

// Reads a CSV file and returns a DataFrame, handling any errors that may occur.
fn read_csv<P: AsRef<Path>>(path: P) -> Result<DataFrame, PolarsError> {
    let file = File::open(path).expect("Cannot open file.");
    CsvReader::new(file).has_header(true).finish()
}

// Separates the given DataFrame into feature and target DataFrames
// Splits the input DataFrame into separate feature and target DataFrames.
fn feature_and_target(df: &DataFrame) -> (Result<DataFrame, PolarsError>, Result<DataFrame, PolarsError>) {
    let features = df.select(vec!["YearsExperience"]);
    let target = df.select(vec!["Salary"]);
    (features, target)
}

// Converts a DataFrame into a DenseMatrix for use in linear regression.
fn convert_to_dense_matrix(df: &DataFrame) -> Result<DenseMatrix<f64>, PolarsError> {
    let (nrows, ncols) = (df.height(), df.width());
    let features_array = df.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    let mut matrix = DenseMatrix::zeros(nrows, ncols);
    let mut row = 0;
    let mut col = 0;

    for &value in features_array.iter() {
        matrix.set(usize::try_from(row).unwrap(), usize::try_from(col).unwrap(), value);

        if col == ncols - 1 {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    Ok(matrix)
}

fn print_evaluation_results(y_test: &[f64], predictions: &[f64], mse: f64, mae: f64, rmse: f64) {
    println!("Model Evaluation Metrics:");
    println!("Mean Squared Error (MSE): {:.4}", mse);
    println!("Mean Absolute Error (MAE): {:.4}", mae);
    println!("Root Mean Squared Error (RMSE): {:.4}\n", rmse);

    println!("{:>10} | {:>10}", "Actual", "Predicted");
    println!("{:->22}", ""); // Prints a line of dashes for separation

    for (actual, predicted) in y_test.iter().zip(predictions.iter()) {
        println!("{:10.2} | {:10.2}", actual, predicted);
    }
}


fn main() {
    let ifile = "C:\\Repos\\Rust_ML\\simple_lr\\data\\Salary_Data.csv";
    let df = match read_csv(&ifile) {
        Ok(df) => df,
        Err(e) => panic!("{:?}", e),
    };

    let (features, target) = match feature_and_target(&df) {
        (Ok(features), Ok(target)) => (features, target),
        (Err(e), _) | (_, Err(e)) => panic!("{:?}", e),
    };

    let features_matrix = match convert_to_dense_matrix(&features) {
        Ok(features_matrix) => Some(features_matrix),
        Err(e) => panic!("{:?}", e),
    };
    let target_array = target.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();

    let y: Vec<f64> = target_array.iter().copied().collect();

    let (x_train, x_test, y_train, y_test) = train_test_split(&features_matrix.unwrap(), &y, 0.3, true);

    let linear_regression = LinearRegression::fit(&x_train, &y_train, Default::default()).unwrap();

    let preds = linear_regression.predict(&x_test).unwrap();

    let mse = mean_squared_error(&y_test, &preds);
    let mae = mean_absolute_error(&y_test, &preds);
    let rmse = mse.sqrt();

    print_evaluation_results(&y_test, &preds, mse, mae, rmse);
}
