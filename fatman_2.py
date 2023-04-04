import time
from datetime import datetime, timedelta

import pandas as pd
from pyspark.sql import functions as sf

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days
# In 13 months (entire period for the input data),
# - all accounts with more than this many incoming "OR" outgoing connections are dropped
MAX_CENTRALITY = 5000
HIGH_RISK_COUNTRIES = [
    "AD",
    "AE",
    "AF",
    "AG",
    "AI",
    "AL",
    "AS",
    "AW",
    "BB",
    "BF",
    "BH",
    "BM",
    "BN",
    "BS",
    "BW",
    "BZ",
    "CH",
    "CK",
    "CR",
    "CW",
    "CY",
    "DM",
    "DO",
    "FJ",
    "GD",
    "GG",
    "GH",
    "GI",
    "GU",
    "HK",
    "HT",
    "IM",
    "IQ",
    "IR",
    "JE",
    "JM",
    "KH",
    "KN",
    "KP",
    "KW",
    "KY",
    "LC",
    "LI",
    "LR",
    "LU",
    "MA",
    "MC",
    "MH",
    "MM",
    "MO",
    "MT",
    "MU",
    "MY",
    "NI",
    "OM",
    "PA",
    "PH",
    "PK",
    "PW",
    "QA",
    "SA",
    "SC",
    "SG",
    "SN",
    "SS",
    "SX",
    "SY",
    "TC",
    "TM",
    "TT",
    "UG",
    "UY",
    "VC",
    "VG",
    "VI",
    "VU",
    "WS",
    "YE",
    "ZW",
]


def rename_columns(dataframe, names):
    for name, new_name in names.items():
        dataframe = dataframe.withColumnRenamed(name, new_name)
    return dataframe


def max_timestamp(dt):
    year, month, date = dt.split("-")
    return (datetime(int(year), int(month), int(date)) + timedelta(days=1)).timestamp()


if __name__ == "__main__":
    # Runtime ~3 hours on ml.m5.4xlarge x 10
    LOGGER.info("Starting the `fatman_2` job")

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    data = spark.read.parquet(f"{MAIN_LOCATION}ftm-input")

    connections_incoming = (
        data.select(sf.col("target").alias("node"), sf.col("source").alias("connection")).drop_duplicates().cache()
    )
    LOGGER.info(f"`connections_incoming` count = {connections_incoming.count():,}")
    connections_outgoing = connections_incoming.select(
        sf.col("connection").alias("node"), sf.col("node").alias("connection")
    )
    connections = connections_incoming.union(connections_outgoing).drop_duplicates().cache()
    LOGGER.info(f"`connections` count = {connections.count():,}")
    central_nodes = connections.groupby("node").agg(sf.count("connection").alias("connections"))
    central_nodes = central_nodes.where(central_nodes.connections >= MAX_CENTRALITY).select("node")
    central_nodes = central_nodes.toPandas()["node"].tolist()
    LOGGER.info(f"`central_nodes` count = {len(central_nodes):,}")

    input_filtered = data.where(~data.source.isin(central_nodes)).where(~data.target.isin(central_nodes))
    location = f"{MAIN_LOCATION}ftm-input-filtered"
    input_filtered.repartition("transaction_date").write.partitionBy("transaction_date").mode("overwrite").parquet(
        location
    )
    data = spark.read.parquet(location)
    LOGGER.info(f"`data_filtered` count = {data.count():,}")

    left_columns = {x.name: f"{x.name}_left" for x in data.schema}
    location_joins = f"{MAIN_LOCATION}ftm-joins"
    dates = sorted([str(x) for x in data.select("transaction_date").distinct().toPandas()["transaction_date"].tolist()])
    LOGGER.info(f"`dates` found = {len(dates)} [{min(dates)} -> {max(dates)}]")
    max_date = str(pd.to_datetime(min(dates)).date() + timedelta(days=365))
    for transaction_date in dates:
        if transaction_date > max_date:
            break
        start_time = time.time()
        start_index = dates.index(transaction_date)
        end_index = start_index + WINDOW + 1
        right_dates = dates[start_index:end_index]
        right = spark.read.option("basePath", location).parquet(
            *[f"{location}/transaction_date={x}" for x in right_dates]
        )
        left = rename_columns(right.where(right.transaction_timestamp < max_timestamp(transaction_date)), left_columns)
        join = left.join(right, left.target_left == right.source, "inner")
        join = join.withColumn("delta", join.transaction_timestamp - join.transaction_timestamp_left)
        # TODO: This should be `(join.delta > -1)` instead
        # TODO: `> 0` is to avoid cycles in rare instances -> Implement a better solution
        # TODO: Also have to fix the (simulation based) max flow calculation, if this is to be changed
        join = join.where(join.delta > 0)
        join.write.parquet(f"{location_joins}/staging_date={transaction_date}", mode="overwrite")
        LOGGER.info(f"[{transaction_date}] Ran in {timedelta(seconds=round(time.time() - start_time))}")  # noqa

    spark.catalog.clearCache()

    joins = spark.read.parquet(location_joins).drop("staging_date")

    # Temporal graph creation
    location_nodes_1 = f"{MAIN_LOCATION}ftm-node-1"
    location_nodes_2 = f"{MAIN_LOCATION}ftm-node-2"
    node_columns = [
        "id",
        "source",
        "target",
        "transaction_date",
        "transaction_timestamp",
        "amount",
        "source_bank_id",
        "target_bank_id",
        "source_country",
        "target_country",
        "cash_related",
    ]
    nodes_1 = (
        (
            joins.select(
                sf.col("id_left").alias("id"),
                sf.col("source_left").alias("source"),
                sf.col("target_left").alias("target"),
                sf.col("transaction_timestamp_left").alias("transaction_timestamp"),
                sf.col("amount_left").alias("amount"),
                sf.col("transaction_date_left").alias("transaction_date"),
                sf.col("source_bank_id_left").alias("source_bank_id"),
                sf.col("target_bank_id_left").alias("target_bank_id"),
                sf.col("source_country_left").alias("source_country"),
                sf.col("target_country_left").alias("target_country"),
                sf.col("cash_related_left").alias("cash_related"),
            )
        )
        .select(*node_columns)
        .drop_duplicates(subset=["id"])
    )
    nodes_1.write.mode("overwrite").parquet(location_nodes_1)
    nodes_2 = joins.select(*node_columns).drop_duplicates(subset=["id"])
    nodes_2.write.mode("overwrite").parquet(location_nodes_2)
    nodes_1 = spark.read.parquet(location_nodes_1)
    nodes_2 = spark.read.parquet(location_nodes_2)
    nodes = nodes_1.union(nodes_2).drop_duplicates(subset=["id"])
    nodes = nodes.withColumn("is_cash_deposit", sf.when(sf.col("source").startswith("cash-"), True).otherwise(False))
    nodes = nodes.withColumn(
        "is_hr_deposit", sf.when(sf.col("target_country").isin(HIGH_RISK_COUNTRIES), True).otherwise(False)
    )

    edges = joins.select(
        sf.col("id_left").alias("src"),
        sf.col("id").alias("dst"),
        sf.col("transaction_date_left").alias("src_date"),
        sf.col("transaction_date").alias("dst_date"),
        "delta",
    )

    nodes_location = f"{MAIN_LOCATION}ftm-nodes"
    edges_location = f"{MAIN_LOCATION}ftm-edges"

    nodes = nodes.repartition("transaction_date")
    nodes.write.partitionBy("transaction_date").mode("overwrite").parquet(nodes_location)

    partition_by = ["src_date", "dst_date"]
    edges.repartition(*partition_by).write.partitionBy(*partition_by).mode("overwrite").parquet(edges_location)
