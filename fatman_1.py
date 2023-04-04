from pyspark.sql import functions as sf
from pyspark.sql import types as st

import inference.jobs.utils as ju
import inference.src.settings as s
import inference.src.spark.helpers as sh

LOGGER = s.get_logger(__name__)
BUCKET = "tmnl-prod-data-scientist-sagemaker-data-intermediate"
MAIN_LOCATION = f"s3a://{BUCKET}/community-detection/exploration/"


if __name__ == "__main__":
    # Runtime ~35 minutes on ml.m5.4xlarge x 10
    LOGGER.info("Starting the `fatman_1` job")

    args = ju.parse_job_arguments()
    _, folder = ju.get_input_output_folders(args)

    settings, spark = ju.setup_job("staging", args.pipeline_prefix_path, args.execution_id)

    # NOTE: This should generate data for 1 year + 21 (WINDOW setting) days

    branch = "feature/fatman"
    branch_location = f"s3a://{BUCKET}/community-detection/{branch}/"
    tem = spark.read.parquet(f"{branch_location}preprocess/transaction_entity_mapping")
    party = spark.read.parquet(f"{branch_location}staging/dim_party")
    business_party = spark.read.parquet(f"{branch_location}staging/dim_business_party")
    product_account = spark.read.parquet(f"{branch_location}staging/dim_product_account")
    link_party_product_account = spark.read.parquet(f"{branch_location}staging/dim_link_party_product_account")

    output_columns = [
        "id",
        "source",
        "source_bank_id",
        "source_country",
        "target",
        "target_bank_id",
        "target_country",
        "transaction_date",
        "transaction_timestamp",
        "cash_related",
        "amount",
    ]

    LOGGER.info("Loaded input files")

    iban_party_mapping = (
        sh.join(
            product_account,
            link_party_product_account,
            sh.JoinStatement("tmnl_product_record_id"),
            how="inner",
            overwrite_strategy="left",
        )
        .select("account_iban", "tmnl_party_id")
        .dropDuplicates()
        .repartition(16)
    )
    iban_party_mapping = iban_party_mapping.withColumnRenamed("tmnl_party_id", "tmnl_party_id_mapped").cache()
    LOGGER.info(f"`iban_party_mapping` count = {iban_party_mapping.count():,}")

    business_party = (
        business_party.select(
            "tmnl_party_id", "industry_code_1", "industry_code_2", "incorporation_date", "legal_form_desc"
        )
        .distinct()
        .cache()
    )
    LOGGER.info(f"`business_party` count = {business_party.count():,}")

    accounts = party.select("tmnl_party_id", "tmnl_er_id").withColumnRenamed("tmnl_er_id", "er_id").distinct()
    accounts = (
        sh.join(accounts, business_party, sh.JoinStatement("tmnl_party_id"), how="inner", overwrite_strategy="left")
        .repartition(16)
        .cache()
    )
    LOGGER.info(f"`accounts` count = {accounts.count():,}")

    tem = (
        tem.select(
            "tmnl_er_id",
            "tmnl_party_id",
            "bank_id",
            "counter_tmnl_er_id",
            "counter_party_iban",
            "counter_party_noniban",
            "counter_party_bank_id",
            "transaction_date",
            "transaction_time",
            "amount_eur",
            "debit_credit",
            "cash_related",
        )
        .withColumn(
            "counter_party_id",
            sf.when(sf.isnull("counter_party_iban"), sf.col("counter_party_noniban")).otherwise(
                sf.col("counter_party_iban")
            ),
        )
        .withColumn("amount", sf.ceil("amount_eur").cast(st.IntegerType()))
        .drop("counter_party_iban", "counter_party_noniban", "amount_eur")
    )
    # TODO: Why do we have bank id == @
    tem = tem.where(tem.amount > 0).where(tem.bank_id != "@").where(tem.counter_party_bank_id != "@")

    columns_to_drop = ["tmnl_er_id", "counter_tmnl_er_id", "debit_credit"]
    is_non_tmnl = sf.isnull("counter_tmnl_er_id")
    is_tmnl = ~sf.isnull("counter_tmnl_er_id")
    is_credit = tem.debit_credit == "C"
    is_debit = tem.debit_credit == "D"
    is_wire = ~tem.cash_related

    non_tmnl_credits = tem.where(is_non_tmnl & is_credit & is_wire)
    non_tmnl_credits = (
        non_tmnl_credits.withColumnRenamed("counter_party_id", "source")
        .withColumnRenamed("tmnl_party_id", "target")
        .withColumnRenamed("counter_party_bank_id", "source_bank_id")
        .withColumnRenamed("bank_id", "target_bank_id")
        .withColumn("source_country", sf.substring("source", 38, 2))
        .withColumn("target_country", sf.lit("NL"))
        .drop(*columns_to_drop)
        .coalesce(32)
        .cache()
    )
    LOGGER.info(f"`non_tmnl_credits` count = {non_tmnl_credits.count():,}")

    non_tmnl_debits = tem.where(is_non_tmnl & is_debit & is_wire)
    non_tmnl_debits = (
        non_tmnl_debits.withColumnRenamed("tmnl_party_id", "source")
        .withColumnRenamed("counter_party_id", "target")
        .withColumnRenamed("bank_id", "source_bank_id")
        .withColumnRenamed("counter_party_bank_id", "target_bank_id")
        .withColumn("source_country", sf.lit("NL"))
        .withColumn("target_country", sf.substring("source", 38, 2))
        .drop(*columns_to_drop)
        .coalesce(32)
        .cache()
    )
    LOGGER.info(f"`non_tmnl_debits` count = {non_tmnl_debits.count():,}")

    tmnl_transactions = (
        sh.join(
            iban_party_mapping,
            tem.where(is_tmnl & is_credit & is_wire),
            sh.JoinStatement("account_iban", "counter_party_id"),
            how="inner",
            overwrite_strategy="left",
        )
    ).drop("account_iban", "counter_party_id")
    tmnl_transactions = (
        tmnl_transactions.withColumnRenamed("tmnl_party_id_mapped", "source")
        .withColumnRenamed("tmnl_party_id", "target")
        .withColumnRenamed("counter_party_bank_id", "source_bank_id")
        .withColumnRenamed("bank_id", "target_bank_id")
        .withColumn("source_country", sf.lit("NL"))
        .withColumn("target_country", sf.lit("NL"))
        .drop(*columns_to_drop)
        .coalesce(32)
        .cache()
    )
    LOGGER.info(f"`tmnl_transactions` count = {tmnl_transactions.count():,}")

    cash_credits = tem.where(is_credit & tem.cash_related)
    cash_credits = (
        cash_credits.withColumn("source", sf.concat(sf.lit("cash-"), cash_credits.tmnl_party_id))
        .withColumnRenamed("tmnl_party_id", "target")
        .withColumnRenamed("counter_party_bank_id", "source_bank_id")
        .withColumnRenamed("bank_id", "target_bank_id")
        .withColumn("source_country", sf.substring("source", 38, 2))
        .withColumn("target_country", sf.lit("NL"))
        .drop(*columns_to_drop)
        .drop("counter_party_id")
        .coalesce(32)
        .cache()
    )
    LOGGER.info(f"`cash_credits` count = {cash_credits.count():,}")

    cash_debits = tem.where(is_debit & tem.cash_related)
    cash_debits = (
        cash_debits.withColumn("target", sf.concat(sf.lit("cash-"), cash_debits.tmnl_party_id))
        .withColumnRenamed("tmnl_party_id", "source")
        .withColumnRenamed("bank_id", "source_bank_id")
        .withColumnRenamed("counter_party_bank_id", "target_bank_id")
        .withColumn("source_country", sf.lit("NL"))
        .withColumn("target_country", sf.substring("source", 38, 2))
        .drop(*columns_to_drop)
        .drop("counter_party_id")
        .coalesce(32)
        .cache()
    )
    LOGGER.info(f"`cash_debits` count = {cash_debits.count():,}")

    data = (
        non_tmnl_credits.unionByName(non_tmnl_debits)
        .unionByName(tmnl_transactions)
        .unionByName(cash_credits)
        .unionByName(cash_debits)
    )
    data = (
        data.where(data.source != data.target)
        .withColumn(
            "transaction_timestamp",
            sf.unix_timestamp(
                sf.concat("transaction_date", sf.lit("-"), "transaction_time"), format="yyyy-MM-dd-HH:mm:ss"
            ).cast(st.IntegerType()),
        )
        .drop("transaction_time")
    )
    data = data.withColumn("id", sf.monotonically_increasing_id())

    missing_source_country = (data.source_country == "") | data.source_country.isNull()
    missing_target_country = (data.target_country == "") | data.target_country.isNull()

    data = data.withColumn(
        "source_country",
        sf.when(missing_source_country, sf.substring(data.source_bank_id, 5, 2)).otherwise(data.source_country),
    )
    data = data.withColumn(
        "target_country",
        sf.when(missing_target_country, sf.substring(data.target_bank_id, 5, 2)).otherwise(data.target_country),
    )

    non_nullable_columns = ["source", "target", "transaction_date", "transaction_timestamp", "cash_related", "amount"]
    data = data.na.drop(how="any", subset=non_nullable_columns).select(*output_columns)

    # Aggregate transactions happening at the exact timestamp
    data = (
        data.groupby(
            ["source", "target", "transaction_timestamp", "cash_related", "source_bank_id", "target_bank_id"]
        ).agg(
            sf.first("transaction_date").alias("transaction_date"),
            sf.first("id").alias("id"),
            sf.first("source_country").alias("source_country"),
            sf.first("target_country").alias("target_country"),
            sf.sum("amount").alias("amount"),
        )
    ).select(*output_columns)

    output = f"{MAIN_LOCATION}ftm-input"
    data.repartition("transaction_date").write.partitionBy("transaction_date").mode("overwrite").parquet(output)
