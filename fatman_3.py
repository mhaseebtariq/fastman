import time
from datetime import timedelta

from graphframes import GraphFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days
# TODO: Investigate - Spark is doing some funny business with float types!
MULTIPLIER = 100000

if __name__ == "__main__":
    # Runtime ~4.5 hours on ml.m5.4xlarge x 10
    LOGGER.info("Starting the `fatman_3` job")

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    spark.sparkContext.setCheckpointDir(".")

    # Edge weights calculation using 2nd-order graph representation
    nodes_location = f"{MAIN_LOCATION}ftm-nodes"
    edges_location = f"{MAIN_LOCATION}ftm-edges"

    nodes = spark.read.parquet(nodes_location)
    edges = spark.read.parquet(edges_location)

    pattern = "(x1) - [e1] -> (x2)"
    src_second_order = sf.concat(sf.col("x1.source"), sf.lit(">>"), sf.col("x1.target")).alias("src")
    dst_second_order = sf.concat(sf.col("x2.source"), sf.lit(">>"), sf.col("x2.target")).alias("dst")
    dates = edges.groupby(["src_date", "dst_date"]).agg(sf.first("delta")).toPandas()
    dates = [(row["src_date"], row["dst_date"]) for index, row in dates.iterrows()]
    dst_dates = sorted(set([x for _, x in dates]))[:-WINDOW]
    location_co_occurrences = f"{MAIN_LOCATION}ftm-2nd-order-co-occurrences"
    for dst_date in dst_dates:
        start_time = time.time()
        edges_dates = [(x, y) for x, y in dates if y == dst_date]
        nodes_dates = sorted(set([x for y in edges_dates for x in y]))
        edges_locations = [f"{edges_location}/src_date={x}/dst_date={y}" for x, y in edges_dates]
        nodes_locations = [f"{nodes_location}/transaction_date={x}" for x in nodes_dates]
        graph = GraphFrame(spark.read.parquet(*nodes_locations), spark.read.parquet(*edges_locations))
        (
            graph.find(pattern)
            .select(
                src_second_order,
                dst_second_order,
                sf.col("x1.transaction_timestamp").alias("ts"),
                sf.col("x2.id").alias("right"),
            )
            .groupby(["src", "right"])
            .agg(sf.first("dst").alias("dst"), sf.max("ts").alias("moment"))
            .select("src", "dst", "moment")
            .drop_duplicates()
        ).write.mode("overwrite").parquet(f"{location_co_occurrences}/date={dst_date}")
        LOGGER.info(f"[{dst_date}] Ran in {timedelta(seconds=round(time.time() - start_time))}")  # noqa

    spark.catalog.clearCache()

    co_occurrences = spark.read.parquet(location_co_occurrences)
    location_counts = f"{MAIN_LOCATION}ftm-2nd-order-co-o-counts"
    co_occurrences.drop_duplicates().groupby(["src", "dst"]).count().write.mode("overwrite").parquet(location_counts)

    counts_data = spark.read.parquet(location_counts)
    src_perspective = counts_data.groupby("src").agg(sf.sum("count").alias("src_total")).cache()
    LOGGER.info(f"`src_perspective` count = {src_perspective.count():,}")
    dst_perspective = counts_data.groupby("dst").agg(sf.sum("count").alias("dst_total")).cache()
    LOGGER.info(f"`dst_perspective` count = {dst_perspective.count():,}")

    location_weights = f"{MAIN_LOCATION}ftm-2nd-order-weights"
    weights = (
        counts_data.join(src_perspective, "src", "left")
        .withColumn("weight_src", ((sf.col("count") / sf.col("src_total")) * MULTIPLIER).astype(st.IntegerType()))
        .select("src", "dst", "count", "weight_src")
        .join(dst_perspective, "dst", "left")
        .withColumn("weight_dst", ((sf.col("count") / sf.col("dst_total")) * MULTIPLIER).astype(st.IntegerType()))
        .select("src", "dst", "weight_src", "weight_dst")
    ).select("src", "dst", sf.greatest(sf.col("weight_src"), sf.col("weight_dst")).alias("weight"))
    weights.write.mode("overwrite").parquet(location_weights)
