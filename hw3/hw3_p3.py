import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

def main():
    # 1. Initialize SparkSession
    spark = SparkSession.builder \
        .appName("PySparkLogisticRegression") \
        .getOrCreate()

    # 2. Load the input CSV file
    # header=True infers column names, inferSchema=True detects data types
    input_path = "data_logistic_regression.csv"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)

    print("--- Original Data Schema ---")
    df.printSchema()

    # 3. Prepare Features
    # List the names of all your feature columns from the CSV
    feature_columns = ["x1", "x2"]  # Update these to match your CSV headers

    # VectorAssembler combines multiple feature columns into a single vector column named "features"
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    
    dataset = assembler.transform(df)

    # 4. Split data into training and test sets (e.g., 80% train, 20% test)
    train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=42)

    # 5. Initialize and Train the Logistic Regression Model
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=10,
        regParam=0.01          # Regularization parameter
    )

    print("--- Training Model ---")
    lr_model = lr.fit(train_data)

    # 6. Make Predictions on the Test Set
    predictions = lr_model.transform(test_data)

    print("--- Sample Predictions ---")
    predictions.select("prediction", "label", "features").show(5, truncate=False)

    # 7. Evaluate the Model using Area Under ROC
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    auc = evaluator.evaluate(predictions)
    print(f"Test Area Under ROC (AUC): {auc:.4f}")

    # Optional: Print model coefficients and intercept
    print(f"Coefficients: {lr_model.coefficients}")
    print(f"Intercept: {lr_model.intercept}")

    training_summary = lr_model.summary
    
    # Get FPR and TPR from the training/model summary
    fpr = training_summary.rocSelect("FPR").collect()
    tpr = training_summary.rocSelect("TPR").collect()
    
    # Flatten the rows into standard lists
    fpr_list = [row.FPR for row in fpr]
    tpr_list = [row.TPR for row in tpr]
    auc_roc = training_summary.areaUnderROC

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_list, tpr_list, color="darkorange", lw=2, label=f"ROC curve (area = {auc_roc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve.png")  # Saves the plot to disk
    plt.show()                    # Displays the plot interactively

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
