import numpy as np

def insert_random_data(collection):
    num_vectors = 100
    # Generate some random vectors
    vectors = np.random.random((num_vectors, 128)).astype(np.float32)
    ids = [i for i in range(num_vectors)]

    # Insert data into collection
    collection.insert([ids, vectors])
    collection.flush()

def find_random_data(collection, number_of_data):
    query_vectors = np.random.random((1, 128)).astype(np.float32)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(query_vectors, anns_field="vector", param=search_params, limit=number_of_data)
    return results