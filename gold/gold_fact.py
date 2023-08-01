"""
Author : Soumil Nitin Shah
Email shahsoumil519@gmail.com
--additional-python-modules  | faker==11.3.0
--conf  |  spark.serializer=org.apache.spark.serializer.KryoSerializer  --conf spark.sql.hive.convertMetastoreParquet=false --conf spark.sql.hive.convertMetastoreParquet=false --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.hudi.catalog.HoodieCatalog --conf spark.sql.legacy.pathOptionBehavior.enabled=true --conf spark.sql.extensions=org.apache.spark.sql.hudi.HoodieSparkSessionExtension
--datalake-formats | hudi
"""

try:
    from awsglue.transforms import *
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext
    from awsglue.context import GlueContext
    from awsglue.job import Job
    import os, sys, json, uuid, boto3, ast, time, datetime, re
    from dataclasses import dataclass
    from datetime import datetime
    import pandas as pd
    from ast import literal_eval
    from pyspark.sql.functions import lit, udf
    from pyspark.context import SparkContext
    from pyspark.sql.session import SparkSession



except Exception as e:
    print("Error*** : {}".format(e))

args = getResolvedOptions(sys.argv, ["JOB_NAME"])

# Create a Spark session
spark = (SparkSession.builder.config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
         .config('spark.sql.hive.convertMetastoreParquet', 'false') \
         .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.hudi.catalog.HoodieCatalog') \
         .config('spark.sql.extensions', 'org.apache.spark.sql.hudi.HoodieSparkSessionExtension') \
         .config('spark.sql.legacy.pathOptionBehavior.enabled', 'true').getOrCreate())

# Create a Spark context and Glue context
sc = spark.sparkContext
glueContext = GlueContext(sc)
job = Job(glueContext)
logger = glueContext.get_logger()
job.init(args["JOB_NAME"], args)
# ---------------------- Settings --------------------
global BUCKET_NAME
DYNAMODB_LOCK_TABLE_NAME = 'hudi-lock-table'
curr_session = boto3.session.Session()
curr_region = curr_session.region_name
BUCKET_NAME = "hudi-learn-demo-1995"


# ---------------------------------------------------------

def upsert_hudi_table(glue_database, table_name, record_id, precomb_key, table_type, spark_df, partition_fields,
                      enable_partition, enable_cleaner, enable_hive_sync, enable_clustering,
                      enable_meta_data_indexing,
                      use_sql_transformer, sql_transformer_query,
                      target_path, index_type, method='upsert', clustering_column='default'):
    """
    Upserts a dataframe into a Hudi table.

    Args:
        glue_database (str): The name of the glue database.
        table_name (str): The name of the Hudi table.
        record_id (str): The name of the field in the dataframe that will be used as the record key.
        precomb_key (str): The name of the field in the dataframe that will be used for pre-combine.
        table_type (str): The Hudi table type (e.g., COPY_ON_WRITE, MERGE_ON_READ).
        spark_df (pyspark.sql.DataFrame): The dataframe to upsert.
        partition_fields this is used to parrtition data
        enable_partition (bool): Whether or not to enable partitioning.
        enable_cleaner (bool): Whether or not to enable data cleaning.
        enable_hive_sync (bool): Whether or not to enable syncing with Hive.
        use_sql_transformer (bool): Whether or not to use SQL to transform the dataframe before upserting.
        sql_transformer_query (str): The SQL query to use for data transformation.
        target_path (str): The path to the target Hudi table.
        method (str): The Hudi write method to use (default is 'upsert').
        index_type : BLOOM or GLOBAL_BLOOM
    Returns:
        None
    """
    # These are the basic settings for the Hoodie table
    hudi_final_settings = {
        "hoodie.table.name": table_name,
        "hoodie.datasource.write.table.type": table_type,
        "hoodie.datasource.write.operation": method,
        "hoodie.datasource.write.recordkey.field": record_id,
        "hoodie.datasource.write.precombine.field": precomb_key,
    }

    # These settings enable syncing with Hive
    hudi_hive_sync_settings = {
        "hoodie.parquet.compression.codec": "gzip",
        "hoodie.datasource.hive_sync.enable": "true",
        "hoodie.datasource.hive_sync.database": glue_database,
        "hoodie.datasource.hive_sync.table": table_name,
        "hoodie.datasource.hive_sync.partition_extractor_class": "org.apache.hudi.hive.MultiPartKeysValueExtractor",
        "hoodie.datasource.hive_sync.use_jdbc": "false",
        "hoodie.datasource.hive_sync.mode": "hms",
    }

    # These settings enable automatic cleaning of old data
    hudi_cleaner_options = {
        "hoodie.clean.automatic": "true",
        "hoodie.clean.async": "true",
        "hoodie.cleaner.policy": 'KEEP_LATEST_FILE_VERSIONS',
        "hoodie.cleaner.fileversions.retained": "3",
        "hoodie-conf hoodie.cleaner.parallelism": '200',
        'hoodie.cleaner.commits.retained': 5
    }

    # These settings enable partitioning of the data
    partition_settings = {
        "hoodie.datasource.write.partitionpath.field": partition_fields,
        "hoodie.datasource.hive_sync.partition_fields": partition_fields,
        "hoodie.datasource.write.hive_style_partitioning": "true",
    }

    hudi_clustering = {
        "hoodie.clustering.execution.strategy.class": "org.apache.hudi.client.clustering.run.strategy.SparkSortAndSizeExecutionStrategy",
        "hoodie.clustering.inline": "true",
        "hoodie.clustering.plan.strategy.sort.columns": clustering_column,
        "hoodie.clustering.plan.strategy.target.file.max.bytes": "1073741824",
        "hoodie.clustering.plan.strategy.small.file.limit": "629145600"
    }

    # Define a dictionary with the index settings for Hudi
    hudi_index_settings = {
        "hoodie.index.type": index_type,  # Specify the index type for Hudi
    }

    # Define a dictionary with the Fiel Size
    hudi_file_size = {
        "hoodie.parquet.max.file.size": 512 * 1024 * 1024,  # 512MB
        "hoodie.parquet.small.file.limit": 104857600,  # 100MB
    }

    hudi_meta_data_indexing = {
        "hoodie.metadata.enable": "true",
        "hoodie.metadata.index.async": "false",
        "hoodie.metadata.index.column.stats.enable": "true",
        "hoodie.metadata.index.check.timeout.seconds": "60",
        "hoodie.write.concurrency.mode": "optimistic_concurrency_control",
        "hoodie.write.lock.provider": "org.apache.hudi.client.transaction.lock.InProcessLockProvider"
    }

    if enable_meta_data_indexing == True or enable_meta_data_indexing == "True" or enable_meta_data_indexing == "true":
        for key, value in hudi_meta_data_indexing.items():
            hudi_final_settings[key] = value  # Add the key-value pair to the final settings dictionary

    if enable_clustering == True or enable_clustering == "True" or enable_clustering == "true":
        for key, value in hudi_clustering.items():
            hudi_final_settings[key] = value  # Add the key-value pair to the final settings dictionary

    # Add the Hudi index settings to the final settings dictionary
    for key, value in hudi_index_settings.items():
        hudi_final_settings[key] = value  # Add the key-value pair to the final settings dictionary

    for key, value in hudi_file_size.items():
        hudi_final_settings[key] = value  # Add the key-value pair to the final settings dictionary

    # If partitioning is enabled, add the partition settings to the final settings
    if enable_partition == "True" or enable_partition == "true" or enable_partition == True:
        for key, value in partition_settings.items(): hudi_final_settings[key] = value

    # If data cleaning is enabled, add the cleaner options to the final settings
    if enable_cleaner == "True" or enable_cleaner == "true" or enable_cleaner == True:
        for key, value in hudi_cleaner_options.items(): hudi_final_settings[key] = value

    # If Hive syncing is enabled, add the Hive sync settings to the final settings
    if enable_hive_sync == "True" or enable_hive_sync == "true" or enable_hive_sync == True:
        for key, value in hudi_hive_sync_settings.items(): hudi_final_settings[key] = value

    # If there is data to write, apply any SQL transformations and write to the target path
    if spark_df.count() > 0:
        if use_sql_transformer == "True" or use_sql_transformer == "true" or use_sql_transformer == True:
            spark_df.createOrReplaceTempView("temp")
            spark_df = spark.sql(sql_transformer_query)

        spark_df.write.format("hudi"). \
            options(**hudi_final_settings). \
            mode("append"). \
            save(target_path)


class AWSS3(object):
    """Helper class to which add functionality on top of boto3 """

    def __init__(self, bucket):

        self.BucketName = bucket
        self.client = boto3.client("s3")

    def put_files(self, Response=None, Key=None):
        """
        Put the File on S3
        :return: Bool
        """
        try:
            response = self.client.put_object(
                Body=Response, Bucket=self.BucketName, Key=Key
            )
            return "ok"
        except Exception as e:
            raise Exception("Error : {} ".format(e))

    def item_exists(self, Key):
        """Given key check if the items exists on AWS S3 """
        try:
            response_new = self.client.get_object(Bucket=self.BucketName, Key=str(Key))
            return True
        except Exception as e:
            return False

    def get_item(self, Key):

        """Gets the Bytes Data from AWS S3 """

        try:
            response_new = self.client.get_object(Bucket=self.BucketName, Key=str(Key))
            return response_new["Body"].read()

        except Exception as e:
            print("Error :{}".format(e))
            return False

    def find_one_update(self, data=None, key=None):

        """
        This checks if Key is on S3 if it is return the data from s3
        else store on s3 and return it
        """

        flag = self.item_exists(Key=key)

        if flag:
            data = self.get_item(Key=key)
            return data

        else:
            self.put_files(Key=key, Response=data)
            return data

    def delete_object(self, Key):

        response = self.client.delete_object(Bucket=self.BucketName, Key=Key, )
        return response

    def get_all_keys(self, Prefix=""):

        """
        :param Prefix: Prefix string
        :return: Keys List
        """
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.BucketName, Prefix=Prefix)

            tmp = []

            for page in pages:
                for obj in page["Contents"]:
                    tmp.append(obj["Key"])

            return tmp
        except Exception as e:
            return []

    def print_tree(self):
        keys = self.get_all_keys()
        for key in keys:
            print(key)
        return None

    def find_one_similar_key(self, searchTerm=""):
        keys = self.get_all_keys()
        return [key for key in keys if re.search(searchTerm, key)]

    def __repr__(self):
        return "AWS S3 Helper class "


@dataclass
class HUDISettings:
    """Class for keeping track of an item in inventory."""

    table_name: str
    path: str


class HUDIIncrementalReader(AWSS3):
    def __init__(self, bucket, hudi_settings, spark_session):
        AWSS3.__init__(self, bucket=bucket)
        if type(hudi_settings).__name__ != "HUDISettings": raise Exception("please pass correct settings ")
        self.hudi_settings = hudi_settings
        self.spark = spark_session

    def __check_meta_data_file(self):
        """
        check if metadata for table exists
        :return: Bool
        """
        file_name = f"metadata/{self.hudi_settings.table_name}.json"
        return self.item_exists(Key=file_name)

    def __read_meta_data(self):
        file_name = f"metadata/{self.hudi_settings.table_name}.json"

        return ast.literal_eval(self.get_item(Key=file_name).decode("utf-8"))

    def __push_meta_data(self, json_data):
        file_name = f"metadata/{self.hudi_settings.table_name}.json"
        self.put_files(
            Key=file_name, Response=json.dumps(json_data)
        )

    def clean_check_point(self):
        file_name = f"metadata/{self.hudi_settings.table_name}.json"
        self.delete_object(Key=file_name)

    def __get_begin_commit(self):
        self.spark.read.format("hudi").load(self.hudi_settings.path).createOrReplaceTempView("hudi_snapshot")
        commits = list(map(lambda row: row[0], self.spark.sql(
            "select distinct(_hoodie_commit_time) as commitTime from  hudi_snapshot order by commitTime asc").limit(
            50).collect()))

        """begin from start """
        begin_time = int(commits[0]) - 1
        return begin_time

    def __read_inc_data(self, commit_time):
        incremental_read_options = {
            'hoodie.datasource.query.type': 'incremental',
            'hoodie.datasource.read.begin.instanttime': commit_time,
        }
        incremental_df = self.spark.read.format("hudi").options(**incremental_read_options).load(
            self.hudi_settings.path).createOrReplaceTempView("hudi_incremental")

        df = self.spark.sql("select * from  hudi_incremental")

        return df

    def __get_last_commit(self):
        commits = list(map(lambda row: row[0], self.spark.sql(
            "select distinct(_hoodie_commit_time) as commitTime from  hudi_incremental order by commitTime asc").limit(
            50).collect()))
        last_commit = commits[len(commits) - 1]
        return last_commit

    def __run(self):
        """Check the metadata file"""
        flag = self.__check_meta_data_file()
        """if metadata files exists load the last commit and start inc loading from that commit """
        if flag:
            meta_data = json.loads(self.__read_meta_data())
            print(f"""
            ******************LOGS******************
            meta_data {meta_data}
            last_processed_commit : {meta_data.get("last_processed_commit")}
            ***************************************
            """)

            read_commit = str(meta_data.get("last_processed_commit"))
            df = self.__read_inc_data(commit_time=read_commit)

            """if there is no INC data then it return Empty DF """
            if not df.rdd.isEmpty():
                last_commit = self.__get_last_commit()
                self.__push_meta_data(json_data=json.dumps({
                    "last_processed_commit": last_commit,
                    "table_name": self.hudi_settings.table_name,
                    "path": self.hudi_settings.path,
                    "inserted_time": datetime.now().__str__(),

                }))
                return df
            else:
                return df

        else:

            """Metadata files does not exists meaning we need to create  metadata file on S3 and start reading from begining commit"""

            read_commit = self.__get_begin_commit()

            df = self.__read_inc_data(commit_time=read_commit)
            last_commit = self.__get_last_commit()

            self.__push_meta_data(json_data=json.dumps({
                "last_processed_commit": last_commit,
                "table_name": self.hudi_settings.table_name,
                "path": self.hudi_settings.path,
                "inserted_time": datetime.now().__str__(),

            }))

            return df

    def read(self):
        """
        reads INC data and return Spark Df
        :return:
        """

        return self.__run()


def get_spark_df_from_dynamodb_table(dynamodb_table):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(dynamodb_table)
    response = table.scan()

    # Convert DynamoDB items to a list of dictionaries
    items = response['Items']

    # Create a Spark DataFrame from the list of dictionaries
    spark_df = spark.createDataFrame(items)
    return spark_df


def load_hudi_tables(loaders):
    """load Hudi tables """

    for items in loaders.get("source"):
        table_name = items.get("table_name")
        path = items.get("hudi_path")

        if items.get("type") == "FULL":
            spark.read.format("hudi").load(path).createOrReplaceTempView(table_name)

        if items.get("type") == "INC":
            helper = HUDIIncrementalReader(
                bucket=BUCKET_NAME,
                hudi_settings=HUDISettings(
                    table_name=table_name,
                    path=path
                ),
                spark_session=spark
            )
            spark_df = helper.read()
            spark_df.createOrReplaceTempView(table_name)


def main():
    loaders = {
        "source": [
            {
                "table_name": "customers",
                "hudi_path": "s3://hudi-learn-demo-1995/silver/table_name=customers/",
                "type": "FULL"
            },
            {
                "table_name": "orders",
                "hudi_path": "s3://hudi-learn-demo-1995/silver/table_name=orders",
                "type": "INC"
            }
        ]
    }

    load_hudi_tables(loaders=loaders)

    """Business query JOIN GOES HERE"""
    query = """
    SELECT
    c.customer_id,
    c.name,
    c.state,
    c.city,
    c.email,
    o.orderid,
    o.ts as order_ts,
    o.order_value,
    o.priority
FROM
    customers c
JOIN
    orders o
ON
    c.customer_id = o.customer_id;
    """
    spark_df = spark.sql(query)
    print(spark_df.show(2))

    upsert_hudi_table(
        glue_database="hudidb",
        table_name="gold_orders",
        record_id="orderid,customer_id",
        precomb_key="orderid",
        table_type='COPY_ON_WRITE',
        partition_fields="default",
        method='upsert',
        index_type='BLOOM',
        enable_partition=False,
        enable_cleaner=True,
        enable_hive_sync=True,
        enable_clustering='False',
        clustering_column='default',
        enable_meta_data_indexing='true',
        use_sql_transformer=False,
        sql_transformer_query='default',
        target_path="s3://hudi-learn-demo-1995/gold/table_name=gold_orders/",
        spark_df=spark_df,
    )


main()
