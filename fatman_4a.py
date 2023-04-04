import time
from datetime import timedelta

import pandas as pd
from graphframes import GraphFrame
from pyspark.sql import functions as sf

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days
# TODO: Investigate - Spark is doing some funny business with float types!
MULTIPLIER = 100000
MIN_EDGE_WEIGHT = int(0.1 * MULTIPLIER)


if __name__ == "__main__":
    # Runtime ~5 hours on ml.m5.4xlarge x 10
    # TODO: This script can be optimized easily
    LOGGER.info("Starting the `fatman_4a` job")

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    spark.sparkContext.setCheckpointDir(".")

    # Edge weights calculation using 2nd-order graph representation
    nodes_location = f"{MAIN_LOCATION}ftm-nodes"
    edges_location = f"{MAIN_LOCATION}ftm-edges"

    edges = spark.read.parquet(edges_location)

    pattern = "(x1) - [e1] -> (x2)"
    src_second_order = sf.concat(sf.col("x1.source"), sf.lit(">>"), sf.col("x1.target")).alias("src")
    dst_second_order = sf.concat(sf.col("x2.source"), sf.lit(">>"), sf.col("x2.target")).alias("dst")

    src_dates = sorted([x.src_date for x in edges.select("src_date").distinct().collect()])
    LOGGER.info(f"`src_dates` count = {len(src_dates):,}")

    location_weights = f"{MAIN_LOCATION}ftm-2nd-order-weights"
    weights = spark.read.parquet(location_weights)
    weights_filtered = weights.where(sf.col("weight") >= MIN_EDGE_WEIGHT).cache()
    LOGGER.info(f"`weights_filtered` count = {weights_filtered.count():,}")
    # Filter out very weak edges (see MIN_EDGE_WEIGHT)
    edges_staging_location = f"{MAIN_LOCATION}ftm-edges-staging"
    columns = [sf.col("src_id").alias("src"), sf.col("dst_id").alias("dst"), "weight", "delta", "src_date", "dst_date"]
    for src_date in src_dates:
        start_time = time.time()
        src_edges = spark.read.option("basePath", edges_location).parquet(f"{edges_location}/src_date={src_date}")
        nodes_dates = [str(x.date()) for x in pd.date_range(src_date, freq="d", periods=WINDOW + 1)]
        nodes_locations = [f"{nodes_location}/transaction_date={x}" for x in nodes_dates]
        graph = GraphFrame(spark.read.option("basePath", nodes_location).parquet(*nodes_locations), src_edges)
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
                weights_filtered,
                ["src", "dst"],
                "inner",
            )
            .drop("src", "dst")
            .select(*columns)
        ).write.mode("overwrite").parquet(f"{edges_staging_location}/staging_date={src_date}")
        LOGGER.info(f"[{src_date}] Ran in {timedelta(seconds=round(time.time() - start_time))}")  # noqa
