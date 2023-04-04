from pyspark.sql import functions as sf

import inference.jobs.utils as ju
import inference.src.settings as s

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

    edges_staging_location = f"{MAIN_LOCATION}ftm-edges-staging"
    edges_filtered_location = f"{MAIN_LOCATION}ftm-edges-filtered"
    nodes_filtered_location = f"{MAIN_LOCATION}ftm-nodes-filtered"

    nodes = spark.read.parquet(f"{MAIN_LOCATION}ftm-nodes")

    edges_staged = spark.read.parquet(edges_staging_location).drop("staging_date")
    partition_by = ["src_date", "dst_date"]
    edges_staged.repartition(*partition_by).write.partitionBy(*partition_by).mode("overwrite").parquet(
        edges_filtered_location
    )
    LOGGER.info("[Checkpoint] `edges_filtered` saved")
    edges_filtered = spark.read.parquet(edges_filtered_location)

    nodes_1 = edges_filtered.select(sf.col("src").alias("id")).drop_duplicates(subset=["id"]).cache()
    LOGGER.info(f"`nodes_1` count = {nodes_1.count():,}")
    nodes_2 = edges_filtered.select(sf.col("dst").alias("id")).drop_duplicates(subset=["id"]).cache()
    LOGGER.info(f"`nodes_2` count = {nodes_2.count():,}")
    node_ids = nodes_1.union(nodes_2).drop_duplicates(subset=["id"]).cache()
    LOGGER.info(f"`nodes` count = {node_ids.count():,}")

    partition = "transaction_date"
    node_ids.join(nodes, "id", "left").repartition(partition).write.partitionBy(partition).mode("overwrite").parquet(
        nodes_filtered_location
    )
