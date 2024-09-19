from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

def create_index_collection(collection):
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
    collection.create_index(field_name="paper_embedding", index_params=index_params)

def create_collection():
    field_id = FieldSchema(name="paper_id", dtype=DataType.INT64, is_primary=True)
    field_paper = FieldSchema(name="paper_embedding", dtype=DataType.FLOAT_VECTOR, dim=768)

    schema = CollectionSchema(fields=[field_id, field_paper], description="vector db for papers")

    # Create a collection
    collection = Collection(name="papers_collection", schema=schema)
    return collection

def get_collection():
    return Collection(name="papers_collection")