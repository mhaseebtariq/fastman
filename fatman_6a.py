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
            st.StructField("transactions", st.IntegerType(), nullable=False),
            st.StructField("label", st.IntegerType(), nullable=False),
            st.StructField("label_cluster", st.IntegerType(), nullable=False),
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

    @sf.pandas_udf(schema, sf.PandasUDFType.GROUPED_MAP)
    def community_summary(input_data):
        input_data = input_data.sort_values("transaction_timestamp").reset_index(drop=True)
        label, label_cluster = input_data.iloc[0]["label"], input_data.iloc[0]["label_cluster"]
        source_accounts = list(input_data.loc[input_data.source.str.startswith("cash-"), "source"].unique())
        target_accounts = list(
            input_data.loc[input_data.target_country.apply(lambda x: x in HIGH_RISK_COUNTRIES), "target"].unique()
        )
        transactions = simulation(input_data, source_accounts, target_accounts)
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
                    label_cluster,
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
                "label_cluster",
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

    location = f"{MAIN_LOCATION}ftm-window-communities-summarized-a"
    communities.groupby("label").apply(community_summary).write.mode("overwrite").parquet(location)
