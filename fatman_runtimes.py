import time
from datetime import datetime, timedelta

import pandas as pd
from graphframes import GraphFrame
from pyspark.sql import functions as sf

import inference.jobs.utils as ju
import inference.src.settings as s

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
INPUT_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/runtimes/"
WINDOW = 7  # days


def rename_columns(dataframe, names):
    for name, new_name in names.items():
        dataframe = dataframe.withColumnRenamed(name, new_name)
    return dataframe


def max_timestamp(dt):
    year, month, date = dt.split("-")
    return (datetime(int(year), int(month), int(date)) + timedelta(days=1)).timestamp()


if __name__ == "__main__":
    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    # Settings
    start_dates, delta = ["2021-04-01"], 21
    # start_dates, delta = ["2021-04-22", "2021-04-23", "2021-04-24", "2021-04-25"], 1
    # start_dates, delta = ["2021-04-26"], 10
    # start_dates, delta = ["2021-05-06"], 10
    # start_dates, delta = ["2021-05-16"], 30
    # start_dates, delta = ["2021-06-15"], 40
    for start_date in start_dates:
        start_date = pd.to_datetime(start_date).date()
        until = pd.to_datetime(start_date).date() + timedelta(days=delta - 1)
        end_date = (pd.to_datetime(until) + timedelta(days=WINDOW)).date()
        dates_to_process = [str(x.date()) for x in pd.date_range(start_date, until)]
        dst_dates = [str(x.date()) for x in pd.date_range(start_date, end_date)][:-WINDOW]
        LOGGER.info(f"[Dates] = {start_date}, {until}, {end_date}")

        location = f"{INPUT_LOCATION}ftm-input-filtered"
        data = spark.read.parquet(location)
        data = data.where(data.transaction_date >= start_date).where(data.transaction_date <= end_date)
        LOGGER.info(f"`data_filtered` count = {data.count():,}")

        left_columns = {x.name: f"{x.name}_left" for x in data.schema}
        location_joins = f"{MAIN_LOCATION}ftm-joins"
        dates = [str(x.date()) for x in pd.date_range(start_date, end_date)]
        LOGGER.info(f"`dates` found = {len(dates)} [{min(dates)} -> {max(dates)}]")
        temporal_g_time = time.time()
        for transaction_date in dates_to_process:
            start_time = time.time()
            start_index = dates.index(transaction_date)
            end_index = start_index + WINDOW + 1
            right_dates = dates[start_index:end_index]
            right = spark.read.option("basePath", location).parquet(
                *[f"{location}/transaction_date={x}" for x in right_dates]
            )
            left = rename_columns(
                right.where(right.transaction_timestamp < max_timestamp(transaction_date)), left_columns
            )
            join = left.join(right, left.target_left == right.source, "inner")
            join = join.withColumn("delta", join.transaction_timestamp - join.transaction_timestamp_left)
            # TODO: This should be `(join.delta > -1)` instead
            # TODO: `> 0` is to avoid cycles in rare instances -> Implement a better solution
            # TODO: Also have to fix the (simulation based) max flow calculation, if this is to be changed
            join = join.where(join.delta > 0)
            join.write.parquet(f"{location_joins}/staging_date={transaction_date}", mode="overwrite")
            LOGGER.info(f"[{transaction_date}] Ran in {timedelta(seconds=round(time.time() - start_time))}")  # noqa

        spark.catalog.clearCache()

        joins = spark.read.parquet(location_joins).drop("staging_date")

        # Temporal graph creation
        location_nodes_1 = f"{MAIN_LOCATION}ftm-node-1"
        location_nodes_2 = f"{MAIN_LOCATION}ftm-node-2"
        node_columns = [
            "id",
            "source",
            "target",
            "transaction_date",
            "transaction_timestamp",
            "amount",
        ]
        nodes_1 = (
            (
                joins.select(
                    sf.col("id_left").alias("id"),
                    sf.col("source_left").alias("source"),
                    sf.col("target_left").alias("target"),
                    sf.col("transaction_timestamp_left").alias("transaction_timestamp"),
                    sf.col("amount_left").alias("amount"),
                    sf.col("transaction_date_left").alias("transaction_date"),
                )
            )
            .select(*node_columns)
            .drop_duplicates(subset=["id"])
        )
        nodes_1.write.mode("overwrite").parquet(location_nodes_1)
        nodes_2 = joins.select(*node_columns).drop_duplicates(subset=["id"])
        nodes_2.write.mode("overwrite").parquet(location_nodes_2)
        nodes_1 = spark.read.parquet(location_nodes_1)
        nodes_2 = spark.read.parquet(location_nodes_2)
        nodes = nodes_1.union(nodes_2).drop_duplicates(subset=["id"])

        edges = joins.select(
            sf.col("id_left").alias("src"),
            sf.col("id").alias("dst"),
            sf.col("transaction_date_left").alias("src_date"),
            sf.col("transaction_date").alias("dst_date"),
            "delta",
        )

        nodes_location = f"{MAIN_LOCATION}ftm-nodes"
        edges_location = f"{MAIN_LOCATION}ftm-edges"

        nodes = nodes.repartition("transaction_date")
        nodes.write.partitionBy("transaction_date").mode("overwrite").parquet(nodes_location)

        partition_by = ["src_date", "dst_date"]
        edges.repartition(*partition_by).write.partitionBy(*partition_by).mode("overwrite").parquet(edges_location)

        LOGGER.info(f"[StepTime] `temporal graph`: {round(time.time() - temporal_g_time)}")

        spark.sparkContext.setCheckpointDir(".")

        nodes = spark.read.parquet(nodes_location)
        edges = spark.read.parquet(edges_location)

        pattern = "(x1) - [e1] -> (x2)"
        src_second_order = sf.concat(sf.col("x1.source"), sf.lit(">>"), sf.col("x1.target")).alias("src")
        dst_second_order = sf.concat(sf.col("x2.source"), sf.lit(">>"), sf.col("x2.target")).alias("dst")
        dates = edges.groupby(["src_date", "dst_date"]).agg(sf.first("delta")).toPandas()
        dates = [(row["src_date"], row["dst_date"]) for index, row in dates.iterrows()]
        location_co_occurrences = f"{MAIN_LOCATION}ftm-2nd-order-co-occurrences"
        weights_time = time.time()
        for dst_date in dst_dates:
            start_time = time.time()
            dst_date = pd.to_datetime(dst_date).date()
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
        co_occurrences.drop_duplicates().groupby(["src", "dst"]).count().write.mode("overwrite").parquet(
            location_counts
        )

        counts_data = spark.read.parquet(location_counts)
        src_perspective = counts_data.groupby("src").agg(sf.sum("count").alias("src_total")).cache()
        LOGGER.info(f"`src_perspective` count = {src_perspective.count():,}")
        dst_perspective = counts_data.groupby("dst").agg(sf.sum("count").alias("dst_total")).cache()
        LOGGER.info(f"`dst_perspective` count = {dst_perspective.count():,}")

        location_weights = f"{MAIN_LOCATION}ftm-2nd-order-weights"
        weights = (
            counts_data.join(src_perspective, "src", "left")
            .withColumn("weight_src", (sf.col("count") / sf.col("src_total")))
            .select("src", "dst", "count", "weight_src")
            .join(dst_perspective, "dst", "left")
            .withColumn("weight_dst", (sf.col("count") / sf.col("dst_total")))
            .select("src", "dst", "weight_src", "weight_dst")
        ).select("src", "dst", sf.greatest(sf.col("weight_src"), sf.col("weight_dst")).alias("weight"))
        weights.write.mode("overwrite").parquet(location_weights)

        LOGGER.info(f"[StepTime] `2nd-order weights graph`: {round(time.time() - weights_time)}")

        src_second_order = sf.concat(sf.col("x1.source"), sf.lit(">>"), sf.col("x1.target")).alias("src")
        dst_second_order = sf.concat(sf.col("x2.source"), sf.lit(">>"), sf.col("x2.target")).alias("dst")

        location_weights = f"{MAIN_LOCATION}ftm-2nd-order-weights"
        weights = spark.read.parquet(location_weights)
        weights_filtered = weights.where(sf.col("weight") >= 0.1).cache()
        LOGGER.info(f"`weights_filtered` count = {weights_filtered.count():,}")
        # Filter out very weak edges (see MIN_EDGE_WEIGHT)
        apply_weights_time = time.time()
        edges_staging_location = f"{MAIN_LOCATION}ftm-edges-staging"
        columns = [
            sf.col("src_id").alias("src"),
            sf.col("dst_id").alias("dst"),
            "weight",
            "delta",
            "src_date",
            "dst_date",
        ]
        for src_date in dates_to_process:
            start_time = time.time()
            src_edges = spark.read.option("basePath", edges_location).parquet(f"{edges_location}/src_date={src_date}")
            nodes_dates = [str(x.date()) for x in pd.date_range(src_date, freq="d", periods=WINDOW + 1)]
            nodes_locations = [f"{nodes_location}/transaction_date={x}" for x in nodes_dates]
            graph = GraphFrame(spark.read.option("basePath", nodes_location).parquet(*nodes_locations), src_edges)
            (
                graph.find(pattern)
                .select(
                    sf.col("x1.id").alias("src_id"),
                    sf.col("x2.id").alias("dst_id"),
                    sf.col("x1.transaction_date").alias("src_date"),
                    sf.col("x2.transaction_date").alias("dst_date"),
                    sf.col("e1.delta").alias("delta"),
                    src_second_order,
                    dst_second_order,
                )
                .join(
                    weights_filtered,
                    ["src", "dst"],
                    "inner",
                )
                .drop("src", "dst")
                .select(*columns)
            ).write.mode("overwrite").parquet(f"{edges_staging_location}/staging_date={src_date}")
            LOGGER.info(f"[{src_date}] Ran in {timedelta(seconds=round(time.time() - start_time))}")  # noqa

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
        node_ids.join(nodes, "id", "left").repartition(partition).write.partitionBy(partition).mode(
            "overwrite"
        ).parquet(nodes_filtered_location)

        LOGGER.info(f"[StepTime] `Weights apply`: {round(time.time() - apply_weights_time)}")
