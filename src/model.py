from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
from pymilvus import Index

def create_index_collection(collection):
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
    collection.create_index(field_name="vector", index_params=index_params)

def create_collection():
    # Define schema
    field_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    field_vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
    schema = CollectionSchema(fields=[field_id, field_vector], description="vector database example")

    # Create a collection
    collection = Collection(name="example_collection", schema=schema)
    return collection

def get_collection():
    return Collection(name="example_collection")