import json
from collections import Counter

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
    # Runtimes
    # 1: (Inner joins) - Time required to run delta w days + number of days (1 minutes per day)
    # 2: (2nd order weights) Time required to run delta w days + number of days (1.5 minutes per day) + 10 minutes
    # 3: (Applying the weights to T) Time required to run delta w days + number of days (2 minutes per day)
    LOGGER.info("Starting the `fatman_6a` job")

    # Summarize communities of a window

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    locations = [
        f"{MAIN_LOCATION}ftm-communities-exp-dbj-2/hops=2",
        f"{MAIN_LOCATION}ftm-communities-exp-dbj-2/hops=3",
        f"{MAIN_LOCATION}ftm-communities-exp-dbj-2/hops=4",
        f"{MAIN_LOCATION}ftm-communities-exp-dbj-2/hops=5",
        # f"{MAIN_LOCATION}ftm-communities-leiden-wo-weights",
        # f"{MAIN_LOCATION}ftm-communities-leiden-w-weights",
        f"{MAIN_LOCATION}ftm-complete-communities-leiden-wo-weights",
        f"{MAIN_LOCATION}ftm-complete-communities-leiden-w-weights",
    ]
    summaries = [
        f"{MAIN_LOCATION}ftm-communities-summary-exp-dbj-2/hops=2",
        f"{MAIN_LOCATION}ftm-communities-summary-exp-dbj-2/hops=3",
        f"{MAIN_LOCATION}ftm-communities-summary-exp-dbj-2/hops=4",
        f"{MAIN_LOCATION}ftm-communities-summary-exp-dbj-2/hops=5",
        # f"{MAIN_LOCATION}ftm-summary-wo-weights",
        # f"{MAIN_LOCATION}ftm-summary-w-weights",
        f"{MAIN_LOCATION}ftm-complete-summary-wo-weights",
        f"{MAIN_LOCATION}ftm-complete-summary-w-weights",
    ]

    results_all = {}
    dbj_accounts = set()
    unique_parties_found = set()
    for result_loc, summary_loc in zip(locations, summaries):
        name = result_loc[-15:]
        results = spark.read.parquet(result_loc)
        summary = spark.read.parquet(summary_loc)
        if "hops" in name:
            total_processed = summary.count()
        else:
            total_processed = summary.where(summary.max_flow > 0).count()
        summary = summary.where(summary.percentage_forwarded > 0.8).where(summary.percentage_forwarded < 1.2)
        summary = summary.where(summary.max_flow > 10000)
        interesting_cases = summary.count()
        labels = [x.label for x in summary.select("label").distinct().collect()]
        results = results.where(results.label.isin(labels))
        sources = results.select(sf.col("source").alias("id")).drop_duplicates()
        targets = results.select(sf.col("target").alias("id")).drop_duplicates()
        accounts = [x.id for x in sources.union(targets).drop_duplicates().collect() if not x.id.startswith("cash-")]
        number_of_accounts = len(accounts)
        accounts_per_label = (
            results.select(sf.col("source").alias("id"), "label")
            .union(results.select(sf.col("target").alias("id"), "label"))
            .groupby(["id", "label"])
            .agg(sf.countDistinct("id").alias("x"))
            .select(sf.sum("x").alias("y"))
            .collect()[0]
            .y
        )
        overlap_factor = accounts_per_label / number_of_accounts
        print(
            f"[{name}] Total: {total_processed:,} "
            f"| Interesting: {interesting_cases:,} "
            f"| Accounts: {number_of_accounts} "
            f"| Overlap Factor: {overlap_factor}"
        )
        if "hops" in name:
            dbj_accounts = dbj_accounts.union(accounts)
            results_all[name] = dbj_accounts
        else:
            results_all[name] = accounts
        unique_parties_found = unique_parties_found.union(accounts)

    print("\n")
    total = len(unique_parties_found)
    for result_loc in locations:
        name = result_loc[-15:]
        accounts = results_all[name]
        common = len(set(unique_parties_found).intersection(accounts))
        print(f"[{name}] {common} / {total} found")

    summary = spark.read.parquet(f"{MAIN_LOCATION}ftm-complete-summary-w-weights")
    dispensers_in_scope = [x.dispensers for x in summary.select("dispensers").collect()]
    sinks_in_scope = [x.sinks for x in summary.select("sinks").collect()]
    all_diameters = Counter([x.in_scope_diameter for x in summary.select("in_scope_diameter").collect()])
    summary = summary.where(summary.percentage_forwarded > 0.8).where(summary.percentage_forwarded < 1.2)
    summary = summary.where(summary.max_flow > 10000)
    cases_diameters = Counter([x.in_scope_diameter for x in summary.select("in_scope_diameter").collect()])
    interesting_cases = summary.count()
    labels = [x.label for x in summary.select("label").distinct().collect()]
    dispensers_cases = [x.dispensers for x in summary.select("dispensers").collect()]
    sinks_cases = [x.sinks for x in summary.select("sinks").collect()]
