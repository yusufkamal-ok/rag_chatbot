import ollama
import pickle
import os
import streamlit as st

# Model
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
VECTOR_DB_FILE = "vector_db.pkl"

# Load database
if os.path.exists(VECTOR_DB_FILE):
    with open(VECTOR_DB_FILE, "rb") as f:
        VECTOR_DB = pickle.load(f)
else:
    st.error("Vector database tidak ditemukan!")
    st.stop()

# cosinus similarity
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)

# fungsi mengambil konteks dari database
def retrieve(query, top_n=20):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = [(chunk, cosine_similarity(query_embedding, embedding), source) for chunk, embedding, source in VECTOR_DB]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit 
st.set_page_config(page_title="Chatbot")
st.title("Chatbot RAG")

st.write("Tanyakan sesuatu seputar Isuzu mobil D-Max, MUX 4x4, ELF NLR, TRAGA, GIGA FVZ")

# Input chat dari user
query = st.text_input("Masukkan pertanyaan Anda:")
if query:
    retrieved_knowledge = retrieve(query)

    sources = set()
    context_text = '\n'.join(f' - {chunk} (Sumber: {source})' for chunk, _, source in retrieved_knowledge)
    
    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_text}
'''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query},
        ],
        stream=True,
    )

    chatbot_response = ""
    for chunk in stream:
        chatbot_response += chunk['message']['content']

    st.subheader("Jawaban Chatbot:")
    st.write(chatbot_response)

    for _, _, source in retrieved_knowledge:
        sources.add(source)
    
    sources_text = ", ".join(sources) if sources else "Tidak ada sumber ditemukan"
    st.markdown(f"**Sumber informasi:** {sources_text}")

st.sidebar.markdown("---")
st.sidebar.write("**Cara Penggunaan:**")
st.sidebar.write("1. Input Pertanyaan seputar Isuzu mobil D-Max, MUX 4x4, ELF NLR,TRAGA, GIGA FVZ")
st.sidebar.write("2. Jalankan chatbot.")
st.sidebar.write("3. Chatbot akan memberikan jawaban")
