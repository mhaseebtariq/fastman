import igraph as ig
import leidenalg as la  # noqa
import pandas as pd
from pyspark.sql import functions as sf

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

    start_date = str(s.MIN_TRX_DATE)
    location_cc = f"{MAIN_LOCATION}connected-components/start_date={start_date}"
    cc = spark.read.parquet(location_cc)
    cc_counts = cc.groupby("component").count().cache()
    LOGGER.info(f"`components` count = {cc_counts.count():,}")
    components = cc_counts.where(sf.col("count") > 1)
    biggest_components = components.where(sf.col("count") > MAX_COMM_SIZE).toPandas()["component"].tolist()
    components = components.where(~components.component.isin(biggest_components))

    location_communities = f"{MAIN_LOCATION}communities/start_date={start_date}"
    cc.join(components, "component", "inner").select("id", sf.col("component").alias("label")).write.mode(
        "overwrite"
    ).parquet(f"{location_communities}/isolated=1/")

    edges_location = f"{MAIN_LOCATION}ftm-edges-filtered/"
    edges_dates = [str(x.date()) for x in sorted(pd.date_range(start_date, periods=WINDOW, freq="d"))]
    edges_locations = [f"{edges_location}src_date={x}/" for x in edges_dates]

    staging_location = f"{MAIN_LOCATION}ftm-edges-window-staging"
    columns = ["src", "dst", "weight"]
    edges = spark.read.option("basePath", edges_location).parquet(*edges_locations).select(*columns)
    remaining = cc.where(cc.component.isin(biggest_components)).select("id").cache()
    LOGGER.info(f"`big_component` count = {remaining.count():,}")
    remaining.select(sf.col("id").alias("src")).join(edges, "src", "inner").write.mode("overwrite").parquet(
        staging_location
    )
    LOGGER.info(f"[Checkpoint] `edges` staged!")

    spark.catalog.clearCache()

    data = pd.read_parquet(staging_location, columns=columns)
    LOGGER.info("Data Loaded")
    data.loc[:, "weight"] /= MULTIPLIER
    weight_range = (data.weight.min(), data.weight.max())
    LOGGER.info(f"[Checkpoint] `weight_range` = {weight_range}")
    LOGGER.info(f"`edges` count = {data.shape[0]:,}")

    graph = ig.Graph.DataFrame(data, use_vids=False, directed=True)
    LOGGER.info("Graph Loaded")
    communities = la.find_partition(
        graph, la.ModularityVertexPartition, weights="weight", n_iterations=5, max_comm_size=MAX_COMM_SIZE
    )
    LOGGER.info("Communities Detected")
    communities_output = graph.get_vertex_dataframe()
    communities_output.loc[:, "label"] = communities.membership
    spark.createDataFrame(communities_output).write.mode("overwrite").parquet(f"{location_communities}/isolated=0/")
