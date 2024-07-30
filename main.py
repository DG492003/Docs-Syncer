import streamlit as st  # for frontend
from PyPDF2 import PdfReader  # for reading files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # for split texts into chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # for generating embeddings
import google.generativeai as genai  # gemini LLM
from langchain.vectorstores import FAISS  # for fast searchable vector store
from langchain_google_genai import ChatGoogleGenerativeAI  # for conversational chat model
from langchain.chains.question_answering import load_qa_chain  # for qa chain
from langchain.prompts import PromptTemplate  # for custom prompt
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Syncer: Get instant Answers from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.
""")

# Load environment variables from the .env file
load_dotenv()
# Access the API key
api_key = os.getenv('API_KEY')


# function to get the text of the pdfs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# convert text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# now we convert it into embeddings
def get_vector_store(text_chunks, apikey):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=apikey)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# pass these embeddings to LLM
def get_conversational_chain():
    # specify how model should be answered
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # specify the model
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    # setting up the prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # create a chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# function to perform operations on user input
def user_input(user_question, api_key):
    # convert into embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    # load embeddings to vector database
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # retrieves the similar documents as per user query
    docs = new_db.similarity_search(user_question)
    # create a chain
    chain = get_conversational_chain()
    # connect the response with a chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    # show the output
    st.write("Reply: ", response["output_text"])


def main():
    st.header("AI Chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:  # Ensure User question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")


if __name__ == "__main__":
    main()
