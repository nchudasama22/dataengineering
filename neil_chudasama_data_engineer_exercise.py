# Databricks notebook source
# MAGIC %md
# MAGIC #### Create Bronze Tables from Volume
# MAGIC Assume this section will be in a pipeline using file trigger so just adding new files or remaking them will trigger the ingestion into the bronze tables.

# COMMAND ----------

# MAGIC %md
# MAGIC Typically the structure of these tables would be determined before hand and the apprioriate permissions set on them.

# COMMAND ----------


investor_path = "/Volumes/test_workspace/default/investment-csv/investor.csv"
investment_path = "/Volumes/test_workspace/default/investment-csv/investment.csv"
transaction_path = "/Volumes/test_workspace/default/investment-csv/transaction.csv"

investor_df = spark.read.option("header", True).csv(investor_path)
investment_df = spark.read.option("header", True).csv(investment_path)
transaction_df = spark.read.option("header", True).csv(transaction_path)


investor_df.write.mode("overwrite").format("delta").saveAsTable("default.bronze_investor")
investment_df.write.mode("overwrite").format("delta").saveAsTable("default.bronze_investment")

# For transaction (fact table) – append new exports
transaction_df.write.mode("append").format("delta").saveAsTable("default.bronze_transaction")


# COMMAND ----------

# MAGIC %md
# MAGIC Note, this reads all the columns as string, inferring schema does not always work if the rows of a column are all NULL

# COMMAND ----------

# MAGIC %md
# MAGIC #### Silver Layer

# COMMAND ----------

# Create Dummy FX Rate Table

from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType
import pandas as pd
import numpy as np


# Create a date range (first of each month for 2024 + 2025)
date_range = pd.date_range(start="2024-01-01", end="2025-12-01", freq="MS")

# Generate dummy FX rates around ~0.85 with small variation
np.random.seed(42)
fx_rates = np.round(0.85 + np.random.uniform(-0.05, 0.05, len(date_range)), 4)

# Build pandas DataFrame
currency_data = pd.DataFrame({
    "currency": ["EUR"] * len(date_range),
    "effective_date": date_range,
    "fx_rate": fx_rates
})

# Define schema
schema = StructType([
    StructField("currency", StringType(), True),
    StructField("effective_date", DateType(), True),
    StructField("fx_rate", DoubleType(), True)
])

# Convert to Spark DataFrame
currency_df = spark.createDataFrame(currency_data, schema=schema)

# Save as Delta table in the default schema
currency_df.write.mode("overwrite").format("delta").saveAsTable("default.currency_table")

# Quick preview
display(currency_df.limit(5))


# COMMAND ----------

from pyspark.sql.functions import col, when, to_date, year, quarter, date_format, abs

# Load bronze tables
bronze_investor = spark.table("default.bronze_investor")
bronze_investment = spark.table("default.bronze_investment")
bronze_transaction = spark.table("default.bronze_transaction")
currency_table = spark.table("default.currency_table")



joined_table = bronze_transaction\
    .join(bronze_investor, "investor_id", "left")\
    .join(bronze_investment, "investment", "left")

silver_transactions = (
    joined_table

    .withColumn("gl_date", to_date(col("gl_date"), "yyyy-MM-dd"))
    .withColumn("amount_original", col("amount").cast("double"))
    .withColumn("entry_date", to_date(col("entry_date"), "yyyy-MM-dd"))
    .withColumn("exit_date", to_date(col("exit_date"), "yyyy-MM-dd"))
    .withColumn("_export_date", to_date(col("_export_date"), "yyyy-MM-dd"))



    .join(
        currency_table,
        (currency_table.currency == joined_table.currency) &
        (col("currency_table.effective_date") == to_date(date_format(col("gl_date"), "yyyy-MM-01"))),
        "left"
    )


    .withColumn("fx_rate_to_gbp", when(col("fx_rate").isNotNull(), col("fx_rate")).otherwise(1.0))
    .withColumn("amount_gbp", abs(col("amount_original")) * col("fx_rate_to_gbp"))


    .withColumn(
        "cash_flow_direction",
        when(col("transaction_type") == "Distribution", lit("inflow"))
        .when(col("transaction_type").isin("Contribution", "Expense"), lit("outflow"))
        .otherwise(lit("unknown"))
    )
    .withColumn("inflow_flag", when(col("cash_flow_direction") == "inflow", lit(True)).otherwise(lit(False)))
    .withColumn("outflow_flag", when(col("cash_flow_direction") == "outflow", lit(True)).otherwise(lit(False)))
)

silver_transactions = silver_transactions.select(
        col("investment").cast("string"),
        col("investor_id").cast("string"),
        col("transaction_id").cast("string"),
        col("fund").cast("string"),
        col("gl_date").cast("date"),
        col("transaction_type").cast("string"),
        col("test_workspace.default.bronze_transaction.currency").alias("currency").cast("string"), 
        col("amount").cast("string"),
        col("_export_date").cast("date"),
        col("investor_type").cast("string"),
        col("entry_date").cast("date"),
        col("exit_date").cast("date"),
        col("amount_original").cast("double"),
        col("effective_date").cast("date"),
        col("fx_rate").cast("double"),
        col("fx_rate_to_gbp").cast("double"),
        col("amount_gbp").cast("double"),
        col("cash_flow_direction").cast("string"),
        col("inflow_flag").cast("boolean"),
        col("outflow_flag").cast("boolean"))
display(silver_transactions)

# Write to Silver
silver_transactions.write.mode("overwrite").format("delta").saveAsTable("default.silver_transactions")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Gold Layer

# COMMAND ----------

# Create Dummy Date Table
import pandas as pd
from pyspark.sql.types import StructType, StructField, DateType, IntegerType, StringType

# Build a date range
date_range = pd.date_range(start="2020-01-01", end="2030-12-31", freq="D")

date_dim_pd = pd.DataFrame({
    "date_key": date_range,
    "year": date_range.year,
    "quarter": date_range.quarter,
    "month": date_range.month,
    "month_name": date_range.strftime("%B"),
    "month_start": date_range.to_period("M").to_timestamp(),
    "quarter_start": date_range.to_period("Q").to_timestamp()
})

schema = StructType([
    StructField("date_key", DateType(), True),
    StructField("year", IntegerType(), True),
    StructField("quarter", IntegerType(), True),
    StructField("month", IntegerType(), True),
    StructField("month_name", StringType(), True),
    StructField("month_start", DateType(), True),
    StructField("quarter_start", DateType(), True)
])

date_dim = spark.createDataFrame(date_dim_pd, schema=schema)

# Save as Delta
date_dim.write.mode("overwrite").format("delta").saveAsTable("default.date_dim")


# COMMAND ----------

from pyspark.sql.functions import sum as _sum

silver_df = spark.table("default.silver_transactions")
date_dim = spark.table("default.date_dim")

# Join with date dimension
joined = silver_df.join(
    date_dim,
    silver_df.gl_date == date_dim.date_key,
    "left"
)

# Aggregate: inflows, outflows, MoM
gold_performance = (
    joined
    .groupBy("fund", "investor_id", "investor_type", "year", "quarter")
    .agg(
        _sum(when(col("inflow_flag") == True, col("amount_gbp"))).alias("total_inflows_gbp"),
        _sum(when(col("outflow_flag") == True, col("amount_gbp"))).alias("total_outflows_gbp")
    )
    .withColumn("multiple_on_money", col("total_inflows_gbp") / col("total_outflows_gbp"))
)

# Save as Gold table
gold_performance.write.mode("overwrite").format("delta").saveAsTable("default.gold_fund_performance")

display(gold_performance.orderBy("fund", "investor_id", "year", "quarter"))


# COMMAND ----------

#### Trend analysis, which would work well in a dashboard also

# COMMAND ----------

display(spark.table("default.gold_fund_performance"))

# COMMAND ----------

