import ollama
import pickle
import os
import pdfplumber

# Model => ollama
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
VECTOR_DB_FILE = "vector_db.pkl"
PDF_FOLDER = "data"

# fungsi read data
def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text: 
                text += extracted_text + "\n"
    return text

# fungsi add informasi ke database
def add_chunk_to_database(chunk, source, vector_db, existing_chunks):
    if chunk.strip() and chunk not in existing_chunks:  
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        vector_db.append((chunk, embedding, source))  
        existing_chunks.add(chunk)

# load database jika ada
if os.path.exists(VECTOR_DB_FILE):
    print("Loading existing database")
    with open(VECTOR_DB_FILE, "rb") as f:
        VECTOR_DB = pickle.load(f)
    existing_chunks = {chunk for chunk, _, _ in VECTOR_DB}
    print(f"Loaded {len(VECTOR_DB)} chunks from {VECTOR_DB_FILE}")
else:
    VECTOR_DB = []
    existing_chunks = set()

# cek folder data
if not os.path.exists(PDF_FOLDER):
    print(f"Folder '{PDF_FOLDER}' tidak ditemukan")
    exit()

# read data
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    print("Tidak ada file di folder.")
    exit()

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    print(f"Processing: {pdf_file}")

    dataset = read_pdf(pdf_path)
    new_chunks = dataset.split("\n")

    for i, chunk in enumerate(new_chunks):
        add_chunk_to_database(chunk, pdf_file, VECTOR_DB, existing_chunks)
        print(f'Added chunk {i+1} from {pdf_file}')

# save database yang diperbarui
with open(VECTOR_DB_FILE, "wb") as f:
    pickle.dump(VECTOR_DB, f)
print(f"Vector database updated and saved {VECTOR_DB_FILE}")

# cosinus similarity
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)

# fungsi mengambil konteks dari database
def retrieve(query, top_n=25):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = [(chunk, cosine_similarity(query_embedding, embedding), source) for chunk, embedding, source in VECTOR_DB]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# chatbot
while True:
    input_query = input("Ask me a question (type 'stop' to exit): ")
    if input_query.lower() == "stop":
        print("Exiting chatbot. Goodbye!")
        break

    retrieved_knowledge = retrieve(input_query)

    sources = set() 
    for chunk, similarity, source in retrieved_knowledge:
        sources.add(source) 

    sources_text = ", ".join(sources) if sources else "No sources found"
    context_text = '\n'.join(f' - {chunk} (Source: {source})' for chunk, _, source in retrieved_knowledge)

    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_text}

If relevant, mention the source of the information.
'''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    print("Chatbot response:")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

    print(f"\n Sources: {sources_text}\n")
