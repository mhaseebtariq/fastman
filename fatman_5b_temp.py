import time

import igraph as ig
import leidenalg as la  # noqa
import numpy as np
import pandas as pd
from pyspark.sql import types as st

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days
MAX_COMM_SIZE = 1000
# TODO: Investigate - Spark is doing some funny business with float types!
MULTIPLIER = 100000


if __name__ == "__main__":
    # Runtime ~40 minutes on ml.r5.12xlarge x 1
    LOGGER.info("Starting the `fatman_5b` job")

    # Community detection on a window

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    start = time.time()
    columns = ["src", "dst", "weight"]
    data = pd.read_parquet(f"{MAIN_LOCATION}ftm-edges-window", columns=columns)
    LOGGER.info("Data Loaded")
    data.loc[:, "weight"] /= MULTIPLIER
    weight_range = (data.weight.min(), data.weight.max())
    LOGGER.info(f"[Checkpoint] `weight_range` = {weight_range}")
    LOGGER.info(f"`edges` count = {data.shape[0]:,}")

    schema = st.StructType(
        [
            st.StructField("id", st.LongType(), nullable=False),
            st.StructField("label", st.LongType(), nullable=False),
        ]
    )

    graph = ig.Graph.DataFrame(data, use_vids=False, directed=True)
    LOGGER.info("Graph Loaded")
    communities = la.find_partition(
        graph, la.ModularityVertexPartition, weights="weight", n_iterations=5, max_comm_size=MAX_COMM_SIZE
    )
    LOGGER.info(f"Communities Detected | {int(time.time() - start)}")
    communities_output = graph.get_vertex_dataframe()
    communities_output.loc[:, "label"] = communities.membership
    communities_output.loc[:, "id"] = communities_output.loc[:, "name"].astype(np.uint64)
    spark.createDataFrame(communities_output.loc[:, ["id", "label"]], schema).write.mode("overwrite").parquet(
        f"{MAIN_LOCATION}ftm-leiden-w-weights"
    )

    start = time.time()
    columns = ["src", "dst"]
    data = pd.read_parquet(f"{MAIN_LOCATION}ftm-edges-window", columns=columns)
    LOGGER.info("Data Loaded")
    LOGGER.info(f"`edges` count = {data.shape[0]:,}")

    graph = ig.Graph.DataFrame(data, use_vids=False, directed=True)
    LOGGER.info("Graph Loaded")
    communities = la.find_partition(graph, la.ModularityVertexPartition, n_iterations=5, max_comm_size=MAX_COMM_SIZE)
    LOGGER.info(f"Communities Detected | {int(time.time() - start)}")
    communities_output = graph.get_vertex_dataframe()
    communities_output.loc[:, "label"] = communities.membership
    communities_output.loc[:, "id"] = communities_output.loc[:, "name"].astype(np.uint64)
    spark.createDataFrame(communities_output.loc[:, ["id", "label"]], schema).write.mode("overwrite").parquet(
        f"{MAIN_LOCATION}ftm-leiden-wo-weights"
    )
