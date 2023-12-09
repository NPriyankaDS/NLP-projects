import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import  CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain


def main():
    st.set_page_config(page_title="Ask your PDF",page_icon=":books:")
    st.title("Ask your PDF:books:")
    st.markdown("This application is used for asking questions to the Pdf document.")

    # Get user's GooglePalm key
    with st.sidebar:
        google_api_key = st.text_input(label = "Google API key", placeholder="Ex sk-2twmA8tfCb8un4...",
        key ="google_api_key_input", help = "How to get a Google api key: Visit https://makersuite.google.com")

        # Container for markdown text

        with st.container():
            st.markdown("""Make sure you have entered your API key.
                        Don't have an API key yet?
                        Read this: Visit https://makersuite.google.com and login with your google account and Get your API key""")

    

    pdf= st.file_uploader("Upload your PDF",type="pdf")
    
    if pdf is not None:
        if not google_api_key:
            st.error("Kindly enter your Google API key")
        else:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            #st.write(text)

            #split into chunks
            text_splitter = CharacterTextSplitter(
                separator = '\n',
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )

            chunks = text_splitter.split_text(text)

            #embedding the chunks 
            embeddings = HuggingFaceEmbeddings(
                model_name = "sentence-transformers/all-mpnet-base-v2",
                model_kwargs = {'device':'cpu'},
                encode_kwargs = {'normalize_embeddings':False}
            )

            knowledge_base = FAISS.from_texts(chunks,embeddings)

            question = st.text_input("Ask a question to your PDF")
            if question:
                with st.spinner("Processing"):
                    docs = knowledge_base.similarity_search(question)
                    llm = GooglePalm(google_api_key=google_api_key,temperature=0.6)
                    chain = load_qa_chain(llm=llm,chain_type = "stuff",verbose=True)
                    response = chain.run(input_documents=docs, question=question)

                    st.success(response)


if __name__ == "__main__":
    main()