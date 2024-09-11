from pymilvus import connections
import model
import repository

# Connect to Milvus server (localhost for Docker installation)
connections.connect(alias="default", host="127.0.0.1", port="19530")

#collection = model.create_collection()
#model.create_index_collection(collection)
collection = model.get_collection()
collection.load()
print("Collection created:", collection.name)
print("Collection schema:", collection.schema)
print(f"Collection data size: {collection.num_entities}")


#repository.insert_random_data(collection)
results = repository.find_random_data(collection, 5)

for result in results[0]:
    print(f"ID: {result.id}, Distance: {result.distance}")