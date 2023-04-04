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


class BankAccount:
    def __init__(self, name):
        self.name = name
        self.received_transactions = pd.DataFrame(columns=["source", "amount"])

    def send(self, amount, consume_from_received_transactions):
        if consume_from_received_transactions:
            self.received_transactions.loc[:, "cumulative"] = self.received_transactions.loc[:, "amount"].cumsum()
            amount_filter = self.received_transactions["cumulative"] > amount
            # Consume by deleting the received transactions
            self.received_transactions = self.received_transactions.loc[amount_filter, :]
            if not self.received_transactions.empty:
                # Update the first (remaining) transaction
                remaining = self.received_transactions.iloc[0]["cumulative"] - amount
                self.received_transactions.loc[self.received_transactions.index[0], "amount"] = remaining
            del self.received_transactions["cumulative"]

    def receive(self, from_account, amount, at):
        self.received_transactions = self.received_transactions.append(
            pd.DataFrame([[from_account, amount]], columns=["source", "amount"], index=[at])
        )


class ProxyBank:
    def __init__(self, sources, targets, flow_seconds):
        self.sources = sources
        self.targets = targets
        self.flow_seconds = flow_seconds

        self.accounts = {}

    def register(self, name):
        exists = self.accounts.get(name)
        if exists:
            LOGGER.info("Bank Account already exists - Skipping registration!")
            return exists
        bank_account = BankAccount(name)
        self.accounts[name] = bank_account
        return bank_account

    def transfer(self, amount, from_account_id, to_account_id, at):
        from_account = self.accounts[from_account_id]
        to_account = self.accounts[to_account_id]
        sender_has_unlimited_funds = from_account_id in self.sources
        if sender_has_unlimited_funds:
            amount_sender_can_transfer = amount
        elif not from_account.received_transactions.empty:
            to_expire = from_account.received_transactions.index < (at - self.flow_seconds)
            # Expire, by deleting, old (received) transactions
            from_account.received_transactions = from_account.received_transactions.loc[~to_expire, :]
            # Check the remaining balance
            sender_balance = from_account.received_transactions.loc[:, "amount"].sum()
            # The portion of transaction that can be moved forward
            amount_sender_can_transfer = min([sender_balance, amount])
        else:
            amount_sender_can_transfer = 0

        from_account.send(amount_sender_can_transfer, not sender_has_unlimited_funds)
        to_account.receive(from_account_id, amount_sender_can_transfer, at)
        return amount_sender_can_transfer > 0

    def simulate(self, transactions):
        transactions.sort_values("transaction_timestamp", inplace=True)
        bank_accounts = set(transactions["source"]).union(transactions["target"])
        for name in bank_accounts:
            self.register(name)
        transactions.loc[:, "involved"] = False
        transactions.loc[:, "dispensed"] = 0
        transactions.loc[:, "sunk"] = 0
        columns = ["source", "target", "amount", "transaction_timestamp"]
        for index, row in transactions.iterrows():
            from_account, to_account, amount, at = [row[x] for x in columns]
            involved = self.transfer(amount, from_account, to_account, at)
            if not involved:
                continue
            transactions.loc[index, "involved"] = involved
            if from_account in self.sources:
                transactions.loc[index, "dispensed"] += row["amount"]
            if to_account in self.targets:
                transactions.loc[index, "sunk"] += row["amount"]
        return transactions.loc[transactions.involved, :]


SCHEMA = st.StructType(
    [
        st.StructField("transactions", st.IntegerType(), nullable=False),
        st.StructField("label", st.LongType(), nullable=False),
        st.StructField("isolated", st.BooleanType(), nullable=False),
        st.StructField("accounts", st.IntegerType(), nullable=False),
        st.StructField("sources", st.IntegerType(), nullable=False),
        st.StructField("targets", st.IntegerType(), nullable=False),
        st.StructField("in_scope_transactions", st.IntegerType(), nullable=False),
        st.StructField("in_scope_accounts", st.IntegerType(), nullable=False),
        st.StructField("in_scope_component_sizes", st.StringType(), nullable=False),
        st.StructField("in_scope_diameter", st.IntegerType(), nullable=False),
        st.StructField("dispensers", st.IntegerType(), nullable=False),
        st.StructField("intermediates", st.IntegerType(), nullable=False),
        st.StructField("sinks", st.IntegerType(), nullable=False),
        st.StructField("dispensed", st.IntegerType(), nullable=False),
        st.StructField("sunk", st.IntegerType(), nullable=False),
        st.StructField("max_flow", st.IntegerType(), nullable=False),
        st.StructField("percentage_forwarded", st.FloatType(), nullable=False),
    ]
)


def simulation(transactions, source_accounts, target_accounts, flow_seconds=FLOW_SECONDS):
    bank = ProxyBank(source_accounts, target_accounts, flow_seconds)
    transactions = bank.simulate(transactions)
    max_flow = 0
    for account in target_accounts:
        max_flow += bank.accounts[account].received_transactions.loc[:, "amount"].sum()
    if not transactions.empty:
        transactions.loc[:, "max_flow"] = max_flow
    return transactions


def community_summary(input_data, details=True):
    input_data = input_data.sort_values("transaction_timestamp").reset_index(drop=True)
    label, isolated = input_data.iloc[0]["label"], False
    source_accounts = list(input_data.loc[input_data["is_cash_deposit"], "source"].unique())
    target_accounts = list(input_data.loc[input_data["is_hr_deposit"], "target"].unique())
    transactions = simulation(input_data, source_accounts, target_accounts)
    if details:
        return transactions

    in_scope_transactions = transactions.shape[0]
    all_accounts = list(set(input_data["source"]).union(input_data["target"]))
    in_scope_accounts = set(transactions["source"]).union(transactions["target"])
    dispensers = set(source_accounts).intersection(transactions["source"])
    intermediates = in_scope_accounts.difference(set(source_accounts).union(target_accounts))
    sinks = set(target_accounts).intersection(transactions["target"])
    dispensed, sunk = transactions["dispensed"].sum(), transactions["sunk"].sum()
    max_flow = 0
    if not transactions.empty:
        max_flow = transactions.iloc[0]["max_flow"]
    percentage_forwarded = sunk / (dispensed or 1)

    in_scope_graph = ig.Graph.DataFrame(transactions.loc[:, ["source", "target"]], use_vids=False, directed=True)
    in_scope_component_sizes = in_scope_graph.connected_components(mode="weak").sizes()
    in_scope_diameter = in_scope_graph.diameter()
    in_scope_diameter = 1 if np.isnan(in_scope_diameter) else in_scope_diameter

    output = pd.DataFrame(
        [
            [
                input_data.shape[0],
                label,
                isolated,
                len(all_accounts),
                len(source_accounts),
                len(target_accounts),
                in_scope_transactions,
                len(in_scope_accounts),
                json.dumps(in_scope_component_sizes),
                in_scope_diameter,
                len(dispensers),
                len(intermediates),
                len(sinks),
                dispensed,
                sunk,
                max_flow,
                percentage_forwarded,
            ]
        ],
        columns=[
            "transactions",
            "label",
            "isolated",
            "accounts",
            "sources",
            "targets",
            "in_scope_transactions",
            "in_scope_accounts",
            "in_scope_component_sizes",
            "in_scope_diameter",
            "dispensers",
            "intermediates",
            "sinks",
            "dispensed",
            "sunk",
            "max_flow",
            "percentage_forwarded",
        ],
    )
    return output


if __name__ == "__main__":
    # Runtime ~10 minutes on ml.m5.4xlarge x 10
    LOGGER.info("Starting the `fatman_6a` job")

    # Summarize communities of a window

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)
    nodes = spark.read.parquet(f"{MAIN_LOCATION}ftm-nodes-window")
    for hop in [2, 3, 4, 5]:
        LOGGER.info(f"[Checkpoint] Processing ho {hop}")
        communities = spark.read.parquet(f"{MAIN_LOCATION}experiments-dbj-communities")
        communities = communities.where(communities.hops == hop).drop("hops")
        communities = communities.join(nodes, "id", "inner").cache()
        LOGGER.info(f"`results` count = {communities.count():,}")

        @sf.pandas_udf(SCHEMA, sf.PandasUDFType.GROUPED_MAP)
        def community_summary_wrapper(input_):
            return community_summary(input_, details=False)

        loc = f"{MAIN_LOCATION}ftm-communities-summary-exp-dbj/hops={hop}"
        communities.groupby("label").apply(community_summary_wrapper).write.mode("overwrite").parquet(loc)
        summary = spark.read.parquet(loc)
        LOGGER.info(f"`summary` count = {summary.count():,}")

        branch = "feature/fatman"
        branch_location = f"s3a://{BUCKET}/community-detection/{branch}/"
        business_party = spark.read.parquet(f"{branch_location}staging/dim_business_party")

        source_features = (
            business_party.join(
                communities.select(sf.col("source").alias("tmnl_party_id")), "tmnl_party_id", how="inner"
            )
            .dropDuplicates()
            .toPandas()
        )
        source_features.columns = [f"source_{x}" for x in source_features.columns]
        LOGGER.info(f"`source_features` count = {source_features.shape[0]:,}")
        source_features = source_features.set_index("source_tmnl_party_id")

        target_features = (
            business_party.join(
                communities.select(sf.col("target").alias("tmnl_party_id")), "tmnl_party_id", how="inner"
            )
            .dropDuplicates()
            .toPandas()
        )
        target_features.columns = [f"target_{x}" for x in target_features.columns]
        LOGGER.info(f"`target_features` count = {target_features.shape[0]:,}")
        target_features = target_features.set_index("target_tmnl_party_id")

        summary = summary.where(summary.max_flow > 10000).toPandas()
        columns = [
            "label",
            "id",
            "transaction_timestamp",
            "source",
            "target",
            "amount",
            "cash_related",
            "dispensed",
            "sunk",
            "max_flow",
            "source_bank_id",
            "source_country",
            "source_incorporation_date",
            "source_industry_code_1",
            "source_industry_code_2",
            "source_legal_form_desc",
            "target_bank_id",
            "target_country",
            "target_incorporation_date",
            "target_industry_code_1",
            "target_industry_code_2",
            "target_legal_form_desc",
        ]

        results = []
        total = summary.shape[0]
        for index, cluster in enumerate(summary.loc[:, "label"]):
            LOGGER.info(f"Processing cluster #{index + 1} of {total}")
            summary = community_summary(communities.where(communities.label == cluster).toPandas(), details=True)
            summary.loc[:, "id"] = summary.loc[:, "id"].astype(int)
            summary = summary.loc[summary["involved"], :].reset_index(drop=True)
            ids = summary.set_index("id").to_dict()
            summary = summary.set_index("source").join(source_features, how="left")
            summary = summary.set_index("target").join(target_features, how="left").reset_index(drop=True)
            summary.loc[:, "source"] = summary.loc[:, "id"].apply(lambda x: ids["source"][x])
            summary.loc[:, "target"] = summary.loc[:, "id"].apply(lambda x: ids["target"][x])
            summary = summary.loc[:, columns]
            results.append(summary)

        results = pd.concat(results, ignore_index=True)
        location = f"{MAIN_LOCATION}ftm-communities-exp-dbj/hops={hop}"
        results.to_parquet(location)
