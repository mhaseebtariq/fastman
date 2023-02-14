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


if __name__ == "__main__":
    LOGGER.info("Starting the `fatman_3` job")

    # Co-occurrence weight for the entire period (1 year)

    nodes_location = f"{MAIN_LOCATION}ftm-nodes/"
    edges_location = f"{MAIN_LOCATION}ftm-edges/"

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    # This is important | Without setting the checkpoint directory, GraphFrames will fail
    spark.sparkContext.setCheckpointDir(".")

    data = spark.read.parquet(f"{MAIN_LOCATION}ftm-input-filtered")
    dates = sorted([str(x) for x in data.select("transaction_date").distinct().toPandas()["transaction_date"].tolist()])
    LOGGER.info(f"`dates` found = {len(dates)} [{min(dates)} -> {max(dates)}]")
    max_date = str(pd.to_datetime(min(dates)).date() + timedelta(days=365))

    pattern = "(x1) - [e1] -> (x2)"
    nodes_days = int(WINDOW * 2)
    location_output = f"{MAIN_LOCATION}ftm-co-occurrence-weights-input/"
    max_node_date = str((pd.to_datetime(max_date) + timedelta(days=WINDOW)).date())
    for start_date in dates:
        if start_date > max_date:
            break
        start_time = time.time()
        nodes_dates = [
            str(x.date()) for x in pd.date_range(start_date, periods=nodes_days, freq="d") if x <= max_node_date
        ]
        nodes_locations = [f"{nodes_location}transaction_date={x}/" for x in nodes_dates]
        day_edges = spark.read.parquet(f"{edges_location}src_date={start_date}/")
        day_nodes = spark.read.parquet(*nodes_locations)
        graph = GraphFrame(day_nodes, day_edges)
        graph.find(pattern).select(
            sf.col("x1.source").alias("start"),
            sf.col("x1.target").alias("middle"),
            sf.col("x2.target").alias("end"),
        ).dropDuplicates().write.mode("overwrite").parquet(f"{location_output}date={start_date}")
        LOGGER.info(f"[{start_date}] Ran in {timedelta(seconds=round(time.time() - start_time))}")  # noqa

    schema = st.StructType(
        [
            st.StructField("src", st.StringType(), nullable=False),
            st.StructField("dst", st.StringType(), nullable=False),
            st.StructField("weight", st.FloatType(), nullable=False),
        ]
    )

    @sf.pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
    def create_connection_edges(input_data):
        end = input_data.iloc[0]["end"]
        input_data = (
            input_data.groupby("start")
            .agg({"middle": "first", "end": "count", "dst_count_per_src": "first"})
            .reset_index()
        )
        input_data.loc[:, "weight"] = input_data.loc[:, "end"] / input_data.loc[:, "dst_count_per_src"]
        input_data.loc[:, "src"] = input_data.start + "-" + input_data.middle
        input_data.loc[:, "dst"] = input_data.middle + f"-{end}"
        return input_data.loc[:, ["src", "dst", "weight"]]

    edges_connections = spark.read.parquet(f"{location_output}")
    dst_counts = edges_connections.groupby("start", "middle").agg(sf.count("end").alias("dst_count_per_src")).cache()
    LOGGER.info(f"{dst_counts.count():,} `sources` found")

    edges_connections = (
        dst_counts.alias("left")
        .join(
            edges_connections.alias("right"),
            (dst_counts.start == edges_connections.start) & (dst_counts.middle == edges_connections.middle),
            "inner",
        )
        .select(sf.col("left.start"), sf.col("left.middle"), sf.col("right.end"), "dst_count_per_src")
        .cache()
    )
    LOGGER.info(f"{edges_connections.count():,} `edges` found")

    location = f"{MAIN_LOCATION}ftm-co-occurrence-weights"
    edges_connections.groupby("middle", "end").apply(create_connection_edges).write.mode("overwrite").parquet(location)
