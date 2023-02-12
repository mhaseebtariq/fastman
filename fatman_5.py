import igraph as ig
import leidenalg as la  # noqa
import pandas as pd

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"


if __name__ == "__main__":
    LOGGER.info("Starting the `fatman_5` job")

    # Community detection on a window

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    location = f"{MAIN_LOCATION}ftm-window-edges-weights"

    data = pd.read_parquet(location, columns=["src", "dst", "weight"])
    LOGGER.info("Data Loaded")
    # This is a bug, this should not happen
    size_before = data.shape[0]
    data = data.loc[data["weight"] > 0, :]
    LOGGER.info(f"Found {size_before - data.shape[0]} edges with negative weights")

    graph = ig.Graph.DataFrame(data, use_vids=False, directed=True)
    LOGGER.info("Graph Loaded")
    communities = la.find_partition(
        graph, la.ModularityVertexPartition, weights="weight", n_iterations=5, max_comm_size=500
    )
    LOGGER.info("Communities Detected")
    communities_output = graph.get_vertex_dataframe()
    communities_output.loc[:, "label"] = communities.membership
    cluster_graph = communities.cluster_graph()
    mapping = dict(zip(cluster_graph.get_vertex_dataframe().index, cluster_graph.clusters().membership))
    communities_output.loc[:, "label_cluster"] = communities_output.loc[:, "label"].apply(mapping.get)
    spark.createDataFrame(communities_output).write.mode("overwrite").parquet(f"{MAIN_LOCATION}ftm-window-communities")
