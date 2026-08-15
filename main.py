import json
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
from numba import jit
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

# go through each jsonl file containing the articles
# go through each line, get the raw text
# make chunks based on 500-10 words (490 words + 10 from the previous chunk)
# save this to a new jsonl file with the text, file number, and title of the wikipage it is found in
# every 10,000 chunks create a new jsonl file (important for embedding later)

"""

Use Cosign similarity to check, idgaf about time, that i have lots. Iterate through all chunks (yes all) and compare using cosign similarity for all embeddings, store the top n (user input) as a list with the index and file number. Then just find the chunk and write to a .txt file.

import numpy as np

# Example: Two 384-dimensional embeddings (simulated here with random data)
embedding1 = np.random.rand(384)
embedding2 = np.random.rand(384)

# 1. Calculate cosine similarity
dot_product = np.dot(embedding1, embedding2)
magnitudes = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
cosine_similarity = dot_product / magnitudes

# 2. Convert to a 0% to 100% scale
# (Cosine similarity ranges from -1 to 1, so we map it to 0 to 100)
similarity_percentage = ((cosine_similarity + 1) / 2) * 100

print(f"Cosine Similarity: {cosine_similarity:.4f}")
print(f"Similarity Score: {similarity_percentage:.2f}%")

"""
print("starting...")

def write_chunk_to_json(chunk, cc, article_name, outfile):
    # print("writing to outfile")
    with open(outfile, "a", encoding="utf-8") as out:
        entry = {
            "from_file": 1,
            "article_name": article_name,
            "chunk_count": cc,
            "chunk": chunk
        }
        out.write(json.dumps(entry) + "\n")

def add_to_chunk_or_word(char, word, chunk, cwi):
    if char != " ":
        word = "".join([word, char])
    else:
        chunk = " ".join([chunk, word])
        word = ""
        cwi += 1
    return word, chunk, cwi

def handle_adding(word, chunk, cwi, previous_text, char, cc, article_name, cwcl, filecount, total_chunk_count):
    word, chunk, cwi = add_to_chunk_or_word(char, word, chunk, cwi)
    if cwi >= cwcl - 26 and cwi < cwcl:
        previous_text = "".join([previous_text, char])

    elif cwi >= cwcl:
        cc += 1
        total_chunk_count += 1
        outfile = "chunks.jsonl"
        write_chunk_to_json(chunk, cc, article_name, outfile)
        chunk = ""
        word = ""
        cwi = 0
    
    return word, chunk, cwi, previous_text, cc, filecount, total_chunk_count

def get_article_length(article):
    words = 0
    characters = 0
    for char in article:
        if char != " ":
            characters += 1
        else:
            words += 1
    return words

def chunking():
    filecount = 1
    total_chunk_count = 0
            
    word = ""
    chunk = ""
    prev_text = ""
    cwi = 0 # current word index
    cc = 0 # chunk count
    cwcl = 256 # chunk word count length

    added_prev = False

    path_of_file = input(" Please enter the path of your file ")

    path_of_file = path_of_file.replace("'", "")


    reader = PdfReader(path_of_file)

    article = ""
    for page in reader.pages:
        article += page.extract_text() + "\n"
    article_name = ""

    article_length = get_article_length(article)
    lcwc = article_length % cwcl # last chunk word count
    blc = (article_length - lcwc) / cwcl # before last chunk 

    print(article_length, lcwc, blc)

    for char in article:
        # print(cc)
        if cc != 0 and added_prev == False:
            chunk = prev_text
            added_prev = True

        if cc < blc:
            # print("Doing chunks that isnt last chunk")
            word, chunk, cwi, prev_text, cc, filecount, total_chunk_count = handle_adding(word, chunk, cwi, prev_text, char, cc, article_name, cwcl, filecount, total_chunk_count)
        elif cc == blc:
            # print("cc == blc")
            # print("doing last chunk")
            word, chunk, cwi, prev_text, cc, filecount, total_chunk_count = handle_adding(word, chunk, cwi, prev_text, char, cc, article_name, lcwc, filecount, total_chunk_count)
            # print(lcwc)
        else:
            print(f"Done   |   Added Chunks: {total_chunk_count}   |   Current File: 1   |   File Count: {filecount}")
            break
    print("Done file")

def embedding(model):
    chunks = []
    with open(f"chunks.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            chunks.append(json.loads(line).get("chunk", "").strip())

    embeddings = model.encode(chunks, batch_size=256, show_progress_bar=True).astype('float32')
    np.save("embeddings.npy", embeddings)
    print(f"done file")

@jit(nopython=True)
def cosine_sim(uq, em):
    assert uq.shape[0] == em.shape[0]
    uqem = np.dot(em, uq)
    uqd = np.dot(uq, uq)
    emd = np.dot(em, em)
    return uqem / np.sqrt(uqd * emd)

def query_and_search(model):
    question_encoding = model.encode(input("What do you want to search? "), show_progress_bar=True)
    
    print("searching......")

    top_10 = [[float('-inf')]] # will have file number and line number

    print(f"going through file")
    vectors = np.load("embeddings.npy", mmap_mode='r')
    for idx, vec in enumerate(vectors):
        cosine = cosine_sim(question_encoding, vec)

        for idxx, val in enumerate(top_10):
            if val[0] < cosine or val[0] == None:
                top_10.insert(idxx, [cosine, idx])
                break
        if len(top_10) > 10:
            top_10.pop()

    print(top_10)
    with open("output.txt", "w", encoding='utf-8') as f:
        pass

    with open("chunks.jsonl", "r", encoding='utf-8') as infile:
        with open("output.txt", "a", encoding='utf-8') as outfile:
            for idx, line in enumerate(infile):
                for val in top_10:
                    if val[1] == idx:
                        print(f"added at idx {idx}")
                        outfile.write(f"{json.loads(line).get("chunk", "").strip()}\n\n")

def handle_drop(event):
    pass

root = TkinterDnD.Tk()
root.title("app")
root.geometry("800x400")





if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunking()
    embedding(model)
    while True:
        todo = input("Do you want to:\n1. Query\n2. Add Another File\n3. Exit?\n\nChoice (1, 2, 3): ")
        if int(todo) == 1:
            query_and_search(model)
        elif int(todo) == 2:
            chunking()
            embedding(model)
        elif int(todo) == 3:
            break
        else:
            print("Invalid Input!")