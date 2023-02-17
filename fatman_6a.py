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
HIGH_RISK_COUNTRIES = [
    "AL",
    "BB",
    "BF",
    "KH",
    "KY",
    "KP",
    "CD",
    "GI",
    "HT",
    "IR",
    "JM",
    "JO",
    "ML",
    "MA",
    "MZ",
    "MM",
    "PA",
    "PH",
    "PK",
    "SN",
    "SS",
    "SY",
    "TZ",
    "TR",
    "UG",
    "AE",
    "YE",
]


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

    def simulate(self, transactions):
        transactions.sort_values("transaction_timestamp", inplace=True)
        bank_accounts = set(transactions["source"]).union(transactions["target"])
        for name in bank_accounts:
            self.register(name)
        columns = ["source", "target", "amount", "transaction_timestamp"]
        for from_account, to_account, amount, at in transactions.loc[:, columns].values:
            self.transfer(amount, from_account, to_account, at)


if __name__ == "__main__":
    # Runtime ~10 minutes on ml.m5.4xlarge x 10
    LOGGER.info("Starting the `fatman_6` job")

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
            st.StructField("label", st.IntegerType(), nullable=False),
            st.StructField("label_cluster", st.IntegerType(), nullable=False),
            st.StructField("diameter", st.IntegerType(), nullable=False),
            st.StructField("in_scope_transactions", st.IntegerType(), nullable=False),
            st.StructField("in_scope_accounts", st.IntegerType(), nullable=False),
            st.StructField("dispensers", st.IntegerType(), nullable=False),
            st.StructField("intermediates", st.IntegerType(), nullable=False),
            st.StructField("sinks", st.IntegerType(), nullable=False),
            st.StructField("dispensed", st.IntegerType(), nullable=False),
            st.StructField("sunk", st.IntegerType(), nullable=False),
            st.StructField("percentage_forwarded", st.FloatType(), nullable=False),
        ]
    )

    def simulation(transactions, source_accounts, target_accounts, flow_seconds=FLOW_SECONDS):
        bank = ProxyBank(source_accounts, target_accounts, flow_seconds)
        bank.simulate(transactions)
        return bank

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
        label, label_cluster = input_data.iloc[0]["label"], input_data.iloc[0]["label_cluster"]
        source_accounts = list(input_data.loc[input_data.source.str.startswith("cash-"), "source"].unique())
        target_accounts = list(
            input_data.loc[input_data.target_country.apply(lambda x: x in HIGH_RISK_COUNTRIES), "target"].unique()
        )
        bank = simulation(input_data, source_accounts, target_accounts)
        all_accounts = set(input_data["source"]).union(input_data["target"])
        in_scope_accounts = set()
        dispensed, sunk = 0, 0
        in_scope_transactions = 0
        for account in all_accounts:
            received = bank.accounts[account].received_transactions
            received = received.loc[received["amount"] > 0, :]
            in_scope_transactions += received.shape[0]
            in_scope_accounts.union(received["source"].tolist())
            received.loc[:, "received_from_source"] = received.loc[:, "source"].apply(lambda x: x in source_accounts)
            dispensed += received.loc[received["received_from_source"], "amount"].sum()
            if account in target_accounts:
                sunk += received.loc[:, "amount"].sum()
        percentage_forwarded = sunk / (dispensed or 1)
        intermediates = in_scope_accounts.difference(set(source_accounts).union(target_accounts))

        output = pd.DataFrame(
            [
                [
                    json.dumps(component_sizes),
                    label,
                    label_cluster,
                    diameter,
                    in_scope_transactions,
                    len(in_scope_accounts),
                    len(source_accounts),
                    len(intermediates),
                    len(target_accounts),
                    dispensed,
                    sunk,
                    percentage_forwarded,
                ]
            ],
            columns=[
                "component_sizes",
                "label",
                "label_cluster",
                "diameter",
                "in_scope_transactions",
                "in_scope_accounts",
                "dispensers",
                "intermediates",
                "sinks",
                "dispensed",
                "sunk",
                "percentage_forwarded",
            ],
        )
        return output

    location = f"{MAIN_LOCATION}ftm-window-communities-summarized-a"
    communities.groupby("label").apply(community_summary).write.mode("overwrite").parquet(location)
