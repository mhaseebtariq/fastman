import time
from datetime import timedelta

import pandas as pd
from graphframes import GraphFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days


def pattern_for(hops):
    last = "x0"
    pattern_constructed = ""
    for x in range(hops):
        current = f"x{x + 1}"
        edge = f"e{x}"
        pattern_constructed += f"({last}) - [{edge}] -> ({current}); "
        last = str(current)
    return pattern_constructed.strip(" ;")


if __name__ == "__main__":
    # Runtime ~5 hours on ml.m5.4xlarge x 10
    # TODO: This script can be optimized easily
    LOGGER.info("Starting the `fatman_4a` job")

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

    nodes = spark.read.parquet(*nodes_locations)
    edges = spark.read.option("basePath", edges_location).parquet(*edges_locations)
    graph = GraphFrame(nodes, edges)

    location = f"{MAIN_LOCATION}experiments"

    start_time = time.time()
    pattern = "(x1) - [e1] -> (x2)"
    (
        graph.find(pattern)
        .where(sf.col("x1.is_cash_deposit") & sf.col("x2.is_hr_deposit"))
        .select(
            sf.array(sf.col("x1.id"), sf.col("x2.id")).alias("ids"),
            sf.col("x1.source").alias("source"),
            sf.col("x2.target").alias("target"),
        )
        .groupby(["source", "target"])
        .agg(sf.collect_list("ids").alias("ids"))
    ).write.parquet(f"{location}/hops=2", mode="overwrite")
    LOGGER.info(f"[2] Ran in {timedelta(seconds=round(time.time() - start_time))}")

    start_time = time.time()
    pattern = "(x1) - [e1] -> (x2); (x2) - [e2] -> (x3)"
    (
        graph.find(pattern)
        .where(sf.col("x1.is_cash_deposit") & sf.col("x3.is_hr_deposit"))
        .select(
            sf.array(sf.col("x1.id"), sf.col("x2.id"), sf.col("x3.id")).alias("ids"),
            sf.col("x1.source").alias("source"),
            sf.col("x3.target").alias("target"),
        )
        .groupby(["source", "target"])
        .agg(sf.collect_list("ids").alias("ids"))
    ).write.parquet(f"{location}/hops=3", mode="overwrite")
    LOGGER.info(f"[3] Ran in {timedelta(seconds=round(time.time() - start_time))}")

    start_time = time.time()
    pattern = "(x1) - [e1] -> (x2); (x2) - [e2] -> (x3); (x3) - [e3] -> (x4)"
    (
        graph.find(pattern)
        .where(sf.col("x1.is_cash_deposit") & sf.col("x4.is_hr_deposit"))
        .select(
            sf.array(sf.col("x1.id"), sf.col("x2.id"), sf.col("x3.id"), sf.col("x4.id")).alias("ids"),
            sf.col("x1.source").alias("source"),
            sf.col("x4.target").alias("target"),
        )
        .groupby(["source", "target"])
        .agg(sf.collect_list("ids").alias("ids"))
    ).write.parquet(f"{location}/hops=4", mode="overwrite")
    LOGGER.info(f"[4] Ran in {timedelta(seconds=round(time.time() - start_time))}")

    start_time = time.time()
    pattern = "(x1) - [e1] -> (x2); (x2) - [e2] -> (x3); (x3) - [e3] -> (x4); (x4) - [e4] -> (x5)"
    (
        graph.find(pattern)
        .where(sf.col("x1.is_cash_deposit") & sf.col("x5.is_hr_deposit"))
        .select(
            sf.array(sf.col("x1.id"), sf.col("x2.id"), sf.col("x3.id"), sf.col("x4.id"), sf.col("x5.id")).alias("ids"),
            sf.col("x1.source").alias("source"),
            sf.col("x5.target").alias("target"),
        )
        .groupby(["source", "target"])
        .agg(sf.collect_list("ids").alias("ids"))
    ).write.parquet(f"{location}/hops=5", mode="overwrite")
    LOGGER.info(f"[5] Ran in {timedelta(seconds=round(time.time() - start_time))}")

    data = spark.read.parquet(location)
    data = data.withColumn("id", sf.monotonically_increasing_id())

    schema = st.StructType(
        [
            st.StructField("id", st.LongType(), nullable=False),
            st.StructField("label", st.LongType(), nullable=False),
            st.StructField("hops", st.IntegerType(), nullable=False),
        ]
    )

    @sf.pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
    def clusters(input_data):
        row = input_data.iloc[0]
        label = row["id"]
        hops = row["hops"]
        ids = list(set([x for y in row["ids"] for x in y]))
        results = pd.DataFrame(ids, columns=["id"])
        results.loc[:, "label"] = label
        results.loc[:, "hops"] = hops
        return results

    data.groupby("id").apply(clusters).write.mode("overwrite").parquet(f"{MAIN_LOCATION}experiments-dbj-communities")
