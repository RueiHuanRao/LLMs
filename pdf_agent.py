# -*- encoding=utf-8 -*-

import os
from getpass import getpass

from langchain.document_loaders import PyPDFDirectoryLoader  # to read pdf files  # noqa
from langchain.text_splitter import RecursiveCharacterTextSplitter  # to tokenise the doc  # noqa
from langchain.embeddings import GPT4AllEmbeddings  # to transform the chunked doc into embedding space  # noqa
from langchain.vectorstores import FAISS, Pinecone
import pinecone  # noqa
from langchain_community.llms import HuggingFaceHub

from langchain.chains import create_qa_with_sources_chain  # noqa
from langchain.chains import ConversationalRetrievalChain
from config import APIKey


os.environ['HUGGINGFACEHUB_API_TOKEN'] = APIKey.HUGGINGFACEHUB_API_TOKEN.value or getpass("HUGGINGFACEHUB_API_TOKEN: ")  # noqa
os.environ["PINECONE_API_KEY"] = APIKey.PINECONE_API_KEY.value or getpass("PINECONE_API_KEY: ")  # noqa


# load docs
pdf_dir_path = r"C:\Users\rueih\Desktop\Rey_GitHub\LangChain\Agent_examples\PDF_Agent\pdf_files"  # noqa
loader = PyPDFDirectoryLoader(pdf_dir_path)
raw_docs = loader.load()

# split the docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # noqa
doc = text_splitter.split_documents(raw_docs)

# embeddings model
embeddings_model = GPT4AllEmbeddings()
# if wanna store the data, set persist_directory="db" to store in "db" folder
faiss = FAISS.from_documents(documents=doc, embedding=embeddings_model)
retriever = faiss.as_retriever(search_kwargs={"k": 3})
print(retriever.search_type)
print(retriever.search_kwargs)
print(retriever.vectorstore)
# ------------------------------------------------------------------------------- #  # noqa
# using Pinecone
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),  # type: ignore
#     environment="gcp-starter"
# )
index_name = "langchain"
pinecone_docsearch = Pinecone.from_documents(doc, embeddings_model, index_name=index_name)  # noqa
retriever = pinecone_docsearch.as_retriever(search_kwargs={"k": 3})
print(retriever.search_type)
print(retriever.search_kwargs)
print(retriever.vectorstore)
# ------------------------------------------------------------------------------- #  # noqa

# choose LLM model
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.5, "max_length": 128})

#
retrieval_qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    return_source_documents=True,
)

# Output
output = retrieval_qa({
    "question": "Tell me about the trolley car event",
    "chat_history": []
})

print(f"Question: {output['question']}")
print(f"Answer: {output['answer']}")  # Pinecone's result is much better than FAISS  # noqa
print(f"Source: {output['source_documents'][0].metadata['source']}")
len(output['source_documents'])


# -------- #
# Method 2 #
# -------- #
from langchain.chains import RetrievalQA  # noqa

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


def process_llm_response(llm_response):  # cite source
    print(f"\n{llm_response['result']}")
    print("Source:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


query = "Explain trolley car event"
llm_response = qa_chain(query)
process_llm_response(llm_response)
qa_chain.retriever.search_type
qa_chain.retriever.vectorstore
qa_chain.retriever.search_kwargs
qa_chain.combine_documents_chain.llm_chain.prompt.template
