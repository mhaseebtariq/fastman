import pandas as pd
from graphframes import GraphFrame
from pyspark.sql import functions as sf

import inference.jobs.utils as ju
import inference.src.settings as s

WINDOW = 21
LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"


if __name__ == "__main__":
    # Runtime ~40 minutes on ml.m5.4xlarge x 10
    # TODO: This script can be optimized easily
    LOGGER.info("Starting the `fatman_4b` job")

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)
    spark.sparkContext.setCheckpointDir(".")

    nodes_location = f"{MAIN_LOCATION}ftm-nodes"
    edges_location = f"{MAIN_LOCATION}ftm-edges"

    start_date = str(s.MIN_TRX_DATE)
    nodes_dates = [str(x.date()) for x in sorted(pd.date_range(start_date, periods=int(WINDOW * 2) + 1, freq="d"))]
    edges_dates = [str(x.date()) for x in sorted(pd.date_range(start_date, periods=WINDOW, freq="d"))]
    nodes_locations = [f"{nodes_location}/transaction_date={x}/" for x in nodes_dates]
    edges_locations = [f"{edges_location}/src_date={x}/" for x in edges_dates]
    nodes = spark.read.option("basePath", nodes_location).parquet(*nodes_locations)
    edges = spark.read.option("basePath", edges_location).parquet(*edges_locations)

    location_weights = f"{MAIN_LOCATION}ftm-2nd-order-weights"
    weights = spark.read.parquet(location_weights)

    dbj = spark.read.parquet(f"{MAIN_LOCATION}experiments-dbj-communities").select("id").distinct().cache()
    LOGGER.info(f"[Checkpoint] `dbj` count = {dbj.count():,}")
    nodes_window = dbj.join(nodes, "id", "left")
    nodes_window.write.partitionBy("transaction_date").mode("overwrite").parquet(f"{MAIN_LOCATION}ftm-nodes-window")
    nodes_window = spark.read.parquet(f"{MAIN_LOCATION}ftm-nodes-window")

    pattern = "(x1) - [e1] -> (x2)"
    graph = GraphFrame(nodes_window, edges)
    edges_window_temp = (
        graph.find(pattern)
        .select(
            sf.col("x1.id").alias("src"),
            sf.col("x2.id").alias("dst"),
            sf.col("x1.transaction_date").alias("src_date"),
            sf.col("x2.transaction_date").alias("dst_date"),
            sf.col("e1.delta").alias("delta"),
        )
        .cache()
    )
    LOGGER.info(f"[Checkpoint] `edges_window_temp` count = {edges_window_temp.count():,}")

    src_second_order = sf.concat(sf.col("x1.source"), sf.lit(">>"), sf.col("x1.target")).alias("src")
    dst_second_order = sf.concat(sf.col("x2.source"), sf.lit(">>"), sf.col("x2.target")).alias("dst")
    columns = [sf.col("src_id").alias("src"), sf.col("dst_id").alias("dst"), "weight", "delta", "src_date", "dst_date"]
    graph = GraphFrame(nodes_window, edges_window_temp)
    partition_by = ["src_date", "dst_date"]
    (
        graph.find(pattern)
        .select(
            sf.col("x1.id").alias("src_id"),
            sf.col("x2.id").alias("dst_id"),
            sf.col("x1.transaction_date").alias("src_date"),
            sf.col("x2.transaction_date").alias("dst_date"),
            sf.col("e1.delta").alias("delta"),
            src_second_order,
            dst_second_order,
        )
        .join(
            weights,
            ["src", "dst"],
            "inner",
        )
        .drop("src", "dst")
        .select(*columns)
    ).repartition(*partition_by).write.partitionBy(*partition_by).mode("overwrite").parquet(
        f"{MAIN_LOCATION}ftm-edges-window"
    )
