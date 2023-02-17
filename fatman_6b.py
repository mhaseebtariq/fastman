import json

import igraph as ig
import numpy as np
import pandas as pd
from pyspark.sql import functions as sf
from pyspark.sql import types as st

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days
FLOW_SECONDS = 21 * 24 * 60 * 60


if __name__ == "__main__":
    # Runtime ~10 minutes on ml.m5.4xlarge x 10
    LOGGER.info("Starting the `fatman_6a` job")

    # Summarize communities of a window

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    communities = spark.read.parquet(f"{MAIN_LOCATION}ftm-window-communities")

    start_date = str(s.MIN_TRX_DATE)
    nodes_days = int(WINDOW * 2)

    nodes_location = f"{MAIN_LOCATION}ftm-nodes/"
    nodes_dates = [str(x.date()) for x in sorted(pd.date_range(start_date, periods=nodes_days, freq="d"))]
    nodes_locations = [f"{nodes_location}transaction_date={x}/" for x in nodes_dates]
    nodes = spark.read.parquet(*nodes_locations)

    results = communities.join(
        nodes,
        communities.name == nodes.id,
        "inner",
    )
    results = results.drop("name")
    location = f"{MAIN_LOCATION}ftm-window-communities-features"
    results.write.mode("overwrite").parquet(location)

    communities = spark.read.parquet(location)
    schema = st.StructType(
        [
            st.StructField("component_sizes", st.StringType(), nullable=False),
            st.StructField("transactions", st.IntegerType(), nullable=False),
            st.StructField("label", st.IntegerType(), nullable=False),
            st.StructField("label_cluster", st.IntegerType(), nullable=False),
            st.StructField("diameter", st.IntegerType(), nullable=False),
            st.StructField("minimum_cycles", st.IntegerType(), nullable=False),
            st.StructField("accounts", st.IntegerType(), nullable=False),
            st.StructField("dispensers", st.IntegerType(), nullable=False),
            st.StructField("intermediates", st.IntegerType(), nullable=False),
            st.StructField("sinks", st.IntegerType(), nullable=False),
            st.StructField("dispensed", st.IntegerType(), nullable=False),
            st.StructField("sunk", st.IntegerType(), nullable=False),
            st.StructField("percentage_forwarded", st.FloatType(), nullable=False),
        ]
    )

    @sf.pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
    def community_summary(input_data):
        input_data = input_data.sort_values("transaction_timestamp").reset_index(drop=True)
        left_side = input_data.copy(deep=True)
        left_side.loc[:, "key"] = list(left_side.loc[:, "target"])
        left_side = left_side.set_index("key")
        right_side = input_data.copy(deep=True)
        right_side.loc[:, "key"] = list(right_side.loc[:, "source"])
        right_side = right_side.set_index("key")
        joins = left_side.join(right_side, on="key", lsuffix="_left", how="inner")
        joins = joins.loc[joins.transaction_timestamp > joins.transaction_timestamp_left, :]
        new_columns = {x: "src_" + x.replace("_left", "") for x in joins.columns if x.endswith("_left")}
        new_columns.update({x: "dst_" + x for x in joins.columns if not x.endswith("_left")})
        joins = joins.rename(columns=new_columns)
        joins = joins.rename(columns={"src_id": "src", "dst_id": "dst"})
        columns = list(joins.columns)
        _ = columns.remove("src")  # noqa
        _ = columns.remove("dst")  # noqa
        columns = ["src", "dst"] + columns
        joins = joins.loc[:, columns]
        graph = ig.Graph.DataFrame(joins, use_vids=False, directed=True)
        diameter = graph.diameter()
        diameter = 1 if np.isnan(diameter) else diameter + 1
        component_sizes = graph.connected_components(mode="weak").sizes()
        communities_output = graph.get_vertex_dataframe()
        communities_output.loc[:, "in_degree"] = graph.degree(mode="in")
        communities_output.loc[:, "out_degree"] = graph.degree(mode="out")
        communities_output = communities_output.set_index("name")
        communities_output = communities_output.join(input_data.set_index("id"), how="inner")
        all_accounts = set(input_data["source"]).union(input_data["target"])
        dispense_transactions = communities_output.loc[communities_output.in_degree == 0, :]
        sink_transactions = communities_output.loc[communities_output.out_degree == 0, :]
        dispensers = set(dispense_transactions.loc[:, "source"])
        sinks = set(dispense_transactions.loc[:, "target"])
        dispensed = dispense_transactions.loc[:, "amount"].sum()
        sunk = sink_transactions.loc[:, "amount"].sum()
        minimum_cycles = len(dispensers.intersection(sinks))
        intermediates = all_accounts.difference(dispensers.union(sinks))
        dispensed = 1 if dispensed < 1 else dispensed
        percentage_forwarded = sunk / (dispensed or 1)
        label, label_cluster = input_data.iloc[0]["label"], input_data.iloc[0]["label_cluster"]
        output = pd.DataFrame(
            [
                [
                    json.dumps(component_sizes),
                    input_data.shape[0],
                    label,
                    label_cluster,
                    diameter,
                    minimum_cycles,
                    len(all_accounts),
                    len(dispensers),
                    len(intermediates),
                    len(sinks),
                    dispensed,
                    sunk,
                    percentage_forwarded,
                ]
            ],
            columns=[
                "component_sizes",
                "transactions",
                "label",
                "label_cluster",
                "diameter",
                "minimum_cycles",
                "accounts",
                "dispensers",
                "intermediates",
                "sinks",
                "dispensed",
                "sunk",
                "percentage_forwarded",
            ],
        )
        return output

    location = f"{MAIN_LOCATION}ftm-window-communities-summarized-b"
    communities.groupby("label").apply(community_summary).write.mode("overwrite").parquet(location)
