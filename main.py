import json
from sentence_transformers import SentenceTransformer
import numpy as np
from pypdf import PdfReader
from numba import jit
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
from urllib.parse import urljoin
from urllib.request import pathname2url
import platform
import subprocess
import ollama
from tkinter import scrolledtext
import threading

print("starting...")

global model
model = SentenceTransformer('all-MiniLM-L6-v2')
MODEL_NAME = "gemma2:2b"

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

def chunking(path_of_file):
    filecount = 1
    total_chunk_count = 0
            
    word = ""
    chunk = ""
    prev_text = ""
    cwi = 0 # current word index
    cc = 0 # chunk count
    cwcl = 256 # chunk word count length

    added_prev = False

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

def embedding():
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

def query_and_search(user_query):
    question_encoding = model.encode(user_query, show_progress_bar=True)
    
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

def open_output():
    path = "output.txt"
    current_os = platform.system()
    try:
        if current_os == "Windows":
            os.startfile(path)
        elif current_os == "Darwin":  # macOS
            subprocess.run(["open", path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", path], check=True)
    except Exception as e:
        print(f"Native open failed: {e}. Falling back to browser URL format...")
        # Emergency backup: Convert path to strict browser URL structure (file:///...)
        file_url = urljoin('file:', pathname2url(path))
        import webbrowser
        webbrowser.open(file_url)


def stream_ai_response(user_query):

    with open("output.txt", "r", encoding='utf-8') as infile:
        content = infile.read()
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    submit_btn.config(state=tk.DISABLED)
    try:
        stream = ollama.generate(
            model=MODEL_NAME, 
            prompt=f"Answer the following question: {user_query}, using only the following information: {content}", 
            stream=True,
            system="You are a cold, purely objective AI. Zero emotion. No extrapolation. Output only direct facts found in the context.",
            options={
                "temperature": 0.0,
                "top_p": 0.1,
                "top_k": 1,
                "repeat_penalty": 1.2
            }
        )
        for chunk in stream:
            text_chunk = chunk["response"]
            output_box.insert(tk.END, text_chunk)
            output_box.see(tk.END)
            root.update_idletasks()
    except Exception as e:
        output_box.insert(tk.END, f"\n\nERROR: {str(e)}")

    finally:
        output_box.config(state=tk.DISABLED)
        submit_btn.config(state=tk.NORMAL)

def start_query_gen_thread():
    threading.Thread(target=send_query, daemon=True).start()


def handle_drop(event):
    file_paths = root.splitlist(event.data)
    if not file_paths: 
        return
    file_path = file_paths[0]
    file_abs_dnd = os.path.abspath(file_path)

    chunking(file_abs_dnd)
    embedding()
    entry_box.config(state="normal")

def send_query():
    user_query = entry_box.get()
    if not user_query:
        return
    query_and_search(user_query)
    open_output()
    entry_box.delete(0, tk.END)
    stream_ai_response(user_query)



if __name__ == "__main__":

    root = TkinterDnD.Tk()
    root.title("app")
    root.geometry("800x400")

    label = tk.Label(root, text="Drag the file you want to query",
                    bg = "black", width=40, height=8)
    label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    label.drop_target_register(DND_FILES)
    label.dnd_bind("<<Drop>>", handle_drop)

    entry_box = tk.Entry(root, width=30, bg="Black", state="disabled", fg="White")
    entry_box.pack(pady=10)
    submit_btn = tk.Button(root, text="submit", command=start_query_gen_thread)
    submit_btn.pack(pady=10)

    output_box = scrolledtext.ScrolledText(
        root, wrap=tk.WORD, font=("Arial", 12), state=tk.DISABLED
    )
    output_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    root.mainloop()