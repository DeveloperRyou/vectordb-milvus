def text_to_embedding(text, model):
    embedding = model.encode(text, show_progress_bar=False)
    return embedding