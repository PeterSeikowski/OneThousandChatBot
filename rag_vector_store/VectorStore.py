import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


class VectorStore:
    """
    A class to manage a vector store for documents.

    This class facilitates the creation and loading of a vector store
    using specified documents and embeddings.

    Attributes:
        embedding (HuggingFaceEmbeddings): Embedding model used to convert text into vector representations.
        persist_directory (str): Directory path for persisting the vector store.
        collection_name (str): Name of the collection within the vector store.
        documents (List[Document]): List of documents to be stored in the vector store.
        vector_store (Chroma or None): Instance of the Chroma vector store. Initialized in `create_vector_store`.
    """

    def __init__(self,
                 documents: list[Document] = None,
                 persist_directory: str = './ChromaDBVectorStore',
                 collection_name: str = 'documents',
                 ):
        """
                Initialize the VectorStore instance.

                Args:
                    documents (List[Document]): A list of Document objects to be stored in the vector store.
                    persist_directory (str, optional): Directory path where the vector store will be persisted.
                                                       Defaults to './ChromaDBVectorStore'.
                    collection_name (str, optional): Name of the collection within the vector store.
                                                     Defaults to 'documents'.
                """
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
            )
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.documents = documents
        self.vector_store = None

    def create_vector_store(self):
        """
                Create or load the vector store.

                If the specified persistence directory exists, the vector store is loaded from it.
                Otherwise, a new vector store is created with the provided documents.
                """
        if os.path.exists(self.persist_directory):
            print(f'--- The vector store already exists Load vector store... ---')
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embedding
                )
            print(f'--- Vectore store loaded. ---')
        else:
            print(f'--- Creating vector store. ---')
            self.vector_store = Chroma.from_documents(documents=self.documents,
                                  embedding=self.embedding,
                                  collection_name=self.collection_name,
                                  persist_directory=self.persist_directory
                                  )
            print(f'--- Vector store created and loaded, directory: {self.persist_directory}. ---')


