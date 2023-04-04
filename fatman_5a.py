import pandas as pd
from graphframes import GraphFrame

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days


if __name__ == "__main__":
    # Runtime ~5 minutes (per window) on ml.m5.4xlarge x 10
    LOGGER.info("Starting the `fatman_5a` job")

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    # This is important | Without setting the checkpoint directory, GraphFrames will fail
    spark.sparkContext.setCheckpointDir(".")

    nodes_location = f"{MAIN_LOCATION}ftm-nodes-filtered"
    edges_location = f"{MAIN_LOCATION}ftm-edges-filtered"

    start_date = str(s.MIN_TRX_DATE)
    nodes_dates = [str(x.date()) for x in sorted(pd.date_range(start_date, periods=int(WINDOW * 2) + 1, freq="d"))]
    edges_dates = [str(x.date()) for x in sorted(pd.date_range(start_date, periods=WINDOW, freq="d"))]
    nodes_locations = [f"{nodes_location}/transaction_date={x}/" for x in nodes_dates]
    edges_locations = [f"{edges_location}/src_date={x}/" for x in edges_dates]

    nodes = spark.read.parquet(*nodes_locations)
    edges = spark.read.option("basePath", edges_location).parquet(*edges_locations)

    graph = GraphFrame(nodes, edges)
    location_cc = f"{MAIN_LOCATION}connected-components/start_date={start_date}"
    graph.connectedComponents().write.mode("overwrite").parquet(location_cc)
