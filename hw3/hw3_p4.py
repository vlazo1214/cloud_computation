
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg, to_timestamp, to_date, date_format

def main():
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("RetailTransactionsAnalytics") \
        .getOrCreate()

    input_path = "transactions.csv"  

    # ==========================================
    # Task A. Data prep (ETL)
    # ==========================================

    # 1. Load the CSV into a Spark DataFrame with header and schema inference on
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(input_path)

    # 2. Create a new column revenue = quantity * price
    df_transformed = df.withColumn("revenue", col("quantity") * col("price"))

    # 3. Convert tx_ts to a timestamp and add tx_date (date only) and tx_month (yyyy-MM)
    df_transformed = df_transformed \
        .withColumn("tx_timestamp", to_timestamp(col("tx_ts"))) \
        .withColumn("tx_date", to_date(col("tx_timestamp"))) \
        .withColumn("tx_month", date_format(col("tx_timestamp"), "yyyy-MM"))

    # ==========================================
    # Task B. Analytics
    # ==========================================

    # 4. Compute daily revenue per store. Return the top 5 store-days by revenue 
    #    (store_id, tx_date, total_revenue)
    print("--- 4. Top 5 Store-Days by Revenue ---")
    top_store_days = df_transformed \
        .groupBy("store_id", "tx_date") \
        .agg(_sum("revenue").alias("total_revenue")) \
        .orderBy(col("total_revenue").desc()) \
        .limit(5)
    
    top_store_days.show(truncate=False)

    # 5. Find the top 5 products by total revenue overall
    print("--- 5. Top 5 Products by Total Revenue ---")
    top_products = df_transformed \
        .groupBy("product") \
        .agg(_sum("revenue").alias("total_revenue")) \
        .orderBy(col("total_revenue").desc()) \
        .limit(5)
    
    top_products.show(truncate=False)

    # 6. Compute, for each category, the average basket size (mean quantity) and total revenue
    print("--- 6. Category Metrics: Average Basket Size & Total Revenue ---")
    category_metrics = df_transformed \
        .groupBy("category") \
        .agg(
            avg("quantity").alias("avg_basket_size"),
            _sum("revenue").alias("total_revenue")
        ) \
        .orderBy(col("total_revenue").desc())
    
    category_metrics.show(truncate=False)

    # Stop the Spark session cleanly
    spark.stop()

if __name__ == "__main__":
    main()

