import pandas as pd
from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM  # noqa
from pyspark.sql import functions as sf
from pyspark.sql import types as st

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
WINDOW = 21  # days


if __name__ == "__main__":
    # Runtime ~2 hours on ml.m5.4xlarge x 10 | with volume_size_in_gb = 500
    LOGGER.info("Starting the `fatman_4` job")

    # Carry-forward weight for a window
    # Plus the final weight, i.e. = co-occurrence weight * carry-forward weight

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    # This is important | Without setting the checkpoint directory, GraphFrames will fail
    spark.sparkContext.setCheckpointDir(".")

    nodes_location = f"{MAIN_LOCATION}ftm-nodes/"
    edges_location = f"{MAIN_LOCATION}ftm-edges/"

    start_date = str(s.MIN_TRX_DATE)
    nodes_days = int(WINDOW * 2)

    nodes_dates = [str(x.date()) for x in sorted(pd.date_range(start_date, periods=nodes_days, freq="d"))]
    nodes_locations = [f"{nodes_location}transaction_date={x}/" for x in nodes_dates]

    edges_locations = []
    for src_date in [str(x.date()) for x in sorted(pd.date_range(start_date, periods=WINDOW, freq="d"))]:
        dst_dates = [str(x.date()) for x in sorted(pd.date_range(src_date, periods=WINDOW, freq="d"))]
        for dst_date in dst_dates:
            edges_locations.append(f"{edges_location}src_date={src_date}/dst_date={dst_date}")

    nodes = spark.read.parquet(*nodes_locations)
    edges = spark.read.parquet(*edges_locations)

    spark.sparkContext.setCheckpointDir(".")

    graph = GraphFrame(nodes, edges)

    message_to_src = sf.array(  # noqa
        sf.concat(AM.dst["source"], sf.lit("-"), AM.dst["target"]),
        AM.dst["id"],
        AM.dst["amount"],
    )
    amount_weights = (
        graph.aggregateMessages(sf.collect_list(AM.msg).alias("records"), sendToSrc=message_to_src)
        .select("id", "records")
        .repartition(1024, "id")
        .cache()
    )
    LOGGER.info(f"{amount_weights.count():,} nodes processed")

    schema = st.StructType(
        [
            st.StructField("src", st.StringType(), nullable=False),
            st.StructField("dst", st.StringType(), nullable=False),
            st.StructField("dst_name", st.StringType(), nullable=False),
            st.StructField("amount_forwarded", st.IntegerType(), nullable=False),
        ]
    )

    @sf.pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
    def unpivot(input_data):
        row = input_data.iloc[0]
        source_id = row["id"]
        result = pd.DataFrame(row["records"].tolist(), columns=["dst_name", "dst", "amount_forwarded"])
        result.loc[:, "src"] = str(source_id)
        result.loc[:, "amount_forwarded"] = result.loc[:, "amount_forwarded"].astype(int)
        mapping = result.groupby("dst_name").agg({"amount_forwarded": sum}).to_dict()["amount_forwarded"]
        result.loc[:, "amount_forwarded"] = result.loc[:, "dst_name"].apply(mapping.get)
        return result.loc[:, ["src", "dst", "dst_name", "amount_forwarded"]]

    location = f"{MAIN_LOCATION}ftm-window-edges-weights-1"
    amount_weights.groupby("id").apply(unpivot).write.mode("overwrite").parquet(location)
    amount_weights.unpersist()
    del amount_weights

    message_to_dst = sf.array(  # noqa
        sf.concat(AM.src["source"], sf.lit("-"), AM.src["target"]),
        AM.src["id"],
        AM.src["amount"],
    )
    amount_weights = (
        graph.aggregateMessages(sf.collect_list(AM.msg).alias("records"), sendToDst=message_to_dst)
        .select("id", "records")
        .repartition(1024, "id")
        .cache()
    )
    LOGGER.info(f"{amount_weights.count():,} nodes processed")

    schema = st.StructType(
        [
            st.StructField("src", st.StringType(), nullable=False),
            st.StructField("dst", st.StringType(), nullable=False),
            st.StructField("src_name", st.StringType(), nullable=False),
            st.StructField("amount_received", st.IntegerType(), nullable=False),
        ]
    )

    @sf.pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
    def unpivot(input_data):
        row = input_data.iloc[0]
        destination_id = row["id"]
        result = pd.DataFrame(row["records"].tolist(), columns=["src_name", "src", "amount_received"])
        result.loc[:, "dst"] = str(destination_id)
        result.loc[:, "amount_received"] = result.loc[:, "amount_received"].astype(int)
        mapping = result.groupby("src_name").agg({"amount_received": sum}).to_dict()["amount_received"]
        result.loc[:, "amount_received"] = result.loc[:, "src_name"].apply(mapping.get)
        return result.loc[:, ["src", "dst", "src_name", "amount_received"]]

    location = f"{MAIN_LOCATION}ftm-window-edges-weights-2"
    amount_weights.groupby("id").apply(unpivot).write.mode("overwrite").parquet(location)

    weights_1 = spark.read.parquet(f"{MAIN_LOCATION}ftm-window-edges-weights-1")
    weights_2 = spark.read.parquet(f"{MAIN_LOCATION}ftm-window-edges-weights-2")
    weights = weights_1.join(weights_2, ["src", "dst"], "inner")
    weights = weights.withColumn(
        "weight_amount",
        sf.when(weights.amount_received > weights.amount_forwarded, 1).otherwise(  # noqa
            weights.amount_received / weights.amount_forwarded
        ),
    ).drop("amount_forwarded", "amount_received")
    location = f"{MAIN_LOCATION}ftm-window-edges-weights-amount"
    weights.write.mode("overwrite").parquet(location)

    weights_amount = spark.read.parquet(f"{MAIN_LOCATION}ftm-window-edges-weights-amount")
    weights_co_occurrence = spark.read.parquet(f"{MAIN_LOCATION}ftm-co-occurrence-weights")
    weights_co_occurrence = (
        weights_co_occurrence.withColumnRenamed("src", "src_name")
        .withColumnRenamed("dst", "dst_name")
        .withColumnRenamed("weight", "weight_co_occurrence")
    )
    weights = weights_amount.join(weights_co_occurrence, ["src_name", "dst_name"], "inner").drop("src_name", "dst_name")
    weights = weights.withColumn("weight", weights.weight_co_occurrence * weights.weight_amount)

    location = f"{MAIN_LOCATION}ftm-window-edges-weights"
    weights.write.mode("overwrite").parquet(location)
