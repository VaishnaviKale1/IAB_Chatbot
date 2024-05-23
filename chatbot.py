import pickle
import time
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = Flask(__name__)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

template = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50-100 words and 4-10 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
{question}
[/INST]
"""


# Store the URLs and conversation chain in a pickle file
def store_conversation_chain(urls, conversation_chain):
    with open("conversation_chain.pkl", "wb") as f:
        pickle.dump((urls, conversation_chain), f)


# Load the URLs and conversation chain from the pickle file
def load_conversation_chain():
    try:
        with open("conversation_chain.pkl", "rb") as f:
            urls, conversation_chain = pickle.load(f)
        return urls, conversation_chain
    except FileNotFoundError:
        return None, None

# Endpoint to submit URLs
@app.route('/submit_urls', methods=['POST'])
def submit_urls():
    urls = request.json.get('urls')
    splits_docs = prepare_docs(urls)
    vectordb = ingest_into_vectordb(splits_docs)
    conversation_chain = get_conversation_chain(vectordb)
    store_conversation_chain(urls, conversation_chain)
    return jsonify({"message": "URLs submitted successfully"})

# Endpoint for question-answering
@app.route('/qa', methods=['POST'])
def qa():
    start = time.perf_counter()
    urls, conversation_chain = load_conversation_chain()
    if urls is None or conversation_chain is None:
        return jsonify({"error": "URLs not submitted yet"}), 400

    user_question = request.json.get('question')
    if any(char.isdigit() for char in user_question) and any(op in user_question for op in ['+', '-', '*', '/', '%']):
        return jsonify({"answer": "I do not have any knowledge of mathematics. Please ask me a different question."})

    response = conversation_chain.invoke({"question": user_question})
    time_taken = time.perf_counter() - start
    print(f'Time taken -> {time.perf_counter() - start}')

    return jsonify({"answer": response['answer'], "timetaken": time_taken})

 # Prepare the documents by loading them from the URLs and splitting them into chunks
def prepare_docs(urls):
    loader = WebBaseLoader(web_paths=[url for url in urls])
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = text_splitter.split_documents(data)
    return split_docs

# Ingest the documents into the vector store
def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

# Create the conversation chain using the vector store
def get_conversation_chain(vectordb):
    llama_llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        temperature=0.75,
        max_tokens=200,
        top_p=1,
        callback_manager=callback_manager,
        n_ctx=3000)

    retriever = vectordb.as_retriever()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = (ConversationalRetrievalChain.from_llm
                          (llm=llama_llm,
                           retriever=retriever,
                           memory=memory,
                           return_source_documents=True))
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain


if __name__ == '__main__':
    app.run(debug=True)
