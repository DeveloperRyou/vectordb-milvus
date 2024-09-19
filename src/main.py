from pymilvus import connections
from sentence_transformers import SentenceTransformer
import model
import repository
import text
import pdf

# Connect to Milvus server (localhost for Docker installation)
connections.connect(alias="default", host="127.0.0.1", port="19530")

collection = model.create_collection()
model.create_index_collection(collection)
#collection = model.get_collection()
collection.load()
print("Collection created:", collection.name)
print("Collection schema:", collection.schema)
print(f"Collection data size: {collection.num_entities}")

#repository.insert_random_data(collection)
#results = repository.find_random_data(collection, 5)
transformer = SentenceTransformer('paraphrase-mpnet-base-v2')

pdf_data = pdf.extract_text_from_pdf("papers/sample1.pdf")
print("Extracted text from PDF:", pdf_data)

pdf_data_embedding = text.text_to_embedding(pdf_data, transformer)
print("Text embedding:", pdf_data_embedding)

repository.insert_data_to_milvus(collection, 1, pdf_data_embedding)