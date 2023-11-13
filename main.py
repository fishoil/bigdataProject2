"""
Author: Long Huang

Project: Big Data manipulation and analysis with Spark

Resources:  pyspark.sql module documentation,
            pyspark.sql.functions module documentation,
            lecture 13 slides,
            Indian man on YouTube for environment setup on Windows 10,
            StackOverflow for general questions

Environment: Windows 10, Python 3.11, Spark 2.4.1, pylint, input files(nodes, and edges)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc

def initialize_spark_session():
    """Initializes a Spark session and returns it."""
    spark = SparkSession \
        .builder \
        .appName("HetioNetAgain") \
        .master("local") \
        .getOrCreate()
    return spark

def compute_drug_associations(spark, nodes_file_path, edges_file_path):
    """Computes the top 5 drugs by number of gene and disease associations."""
    # Load nodes and edges into DataFrames
    nodes_df = spark.read.csv(nodes_file_path, sep="\t", header=True)
    edges_df = spark.read.csv(edges_file_path, sep="\t", header=True)

    # Filter for drugs in nodes
    drugs_df = nodes_df.filter(col("kind") == "Compound")

    # Define the gene and disease association types
    gene_associations = ['CuG', 'CbG', 'CdG']
    disease_associations = ['CtD', 'CpD']

    # Filter the edges for gene and disease associations
    gene_edges_df = edges_df.filter(col("metaedge").isin(gene_associations))
    disease_edges_df = edges_df.filter(col("metaedge").isin(disease_associations))

    # Join edges with the nodes to get the drug names
    drug_gene_df = gene_edges_df.join(drugs_df, gene_edges_df["source"] == drugs_df["id"])
    drug_disease_df = disease_edges_df.join(drugs_df, disease_edges_df["source"] == drugs_df["id"])

    # Compute the number of genes and diseases associated with each drug
    drug_gene_counts = drug_gene_df.groupBy("name").agg(count("target").alias("gene_count"))
    drug_disease_counts = drug_disease_df.groupBy("name").agg(count("target").alias("disease_count"))

    # Join the gene and disease counts on the drug name
    drug_counts_df = drug_gene_counts.join(drug_disease_counts, "name", "outer").fillna(0)

    # Sort by the number of gene associations in descending order and take the top 5
    top_drugs_df = drug_counts_df.orderBy(desc("gene_count")).limit(5)

    return top_drugs_df


def compute_disease_drug_associations(spark, nodes_file_path, edges_file_path):
    """Computes the top 5 diseases by number of drug associations."""
    # Load nodes and edges into DataFrames
    nodes_df = spark.read.csv(nodes_file_path, sep="\t", header=True)
    edges_df = spark.read.csv(edges_file_path, sep="\t", header=True)

    # Filter for diseases in nodes
    diseases_df = nodes_df.filter(col("kind") == "Disease")

    # Define the disease association types
    disease_associations = ['CtD', 'CpD']

    # Filter the edges for disease associations
    disease_edges_df = edges_df.filter(col("metaedge").isin(disease_associations))

    # Join edges with the nodes to get the disease names
    drug_disease_df = disease_edges_df.join(diseases_df, disease_edges_df["target"] == diseases_df["id"])

    # Compute the number of drugs associated with each disease
    disease_drug_counts = drug_disease_df.groupBy("name").agg(count("source").alias("drug_count"))

    # Sort by the number of drug associations in descending order and take the top 5
    top_diseases_df = disease_drug_counts.orderBy(desc("drug_count")).limit(5)

    return top_diseases_df


def compute_top_drugs_by_gene_associations(spark, nodes_file_path, edges_file_path):
    """Computes the top 5 drugs by number of gene associations.
    Same as Q1 but without disease association."""
    # Load nodes and edges into DataFrames
    nodes_df = spark.read.csv(nodes_file_path, sep="\t", header=True)
    edges_df = spark.read.csv(edges_file_path, sep="\t", header=True)

    # Filter for drugs in nodes
    drugs_df = nodes_df.filter(col("kind") == "Compound")

    # Define the gene association types
    gene_associations = ['CuG', 'CbG', 'CdG']

    # Filter the edges for gene associations
    gene_edges_df = edges_df.filter(col("metaedge").isin(gene_associations))

    # Join edges with the nodes to get the drug names
    drug_gene_df = gene_edges_df.join(drugs_df, gene_edges_df["source"] == drugs_df["id"])

    # Compute the number of genes associated with each drug
    drug_gene_counts = drug_gene_df.groupBy("name").agg(count("target").alias("gene_count"))

    # Sort by the number of gene associations in descending order and take the top 5
    top_drugs_df = drug_gene_counts.orderBy(desc("gene_count")).limit(5)

    return top_drugs_df


def main():
    spark = initialize_spark_session()

    nodes_file_path = "C:/Users/Long Huang/PycharmProjects/bigdataProject2/nodes_test.tsv"
    edges_file_path = "C:/Users/Long Huang/PycharmProjects/bigdataProject2/edges_test.tsv"

    top_drugs_df = compute_drug_associations(spark, nodes_file_path, edges_file_path)
    top_drugs_df.show()

    top_diseases_df = compute_disease_drug_associations(spark, nodes_file_path, edges_file_path)
    top_diseases_df.show()

    top_drugs_by_gene_associations_df = compute_top_drugs_by_gene_associations(spark, nodes_file_path, edges_file_path)
    top_drugs_by_gene_associations_df.show()

    spark.stop()

if __name__ == "__main__":
    main()
