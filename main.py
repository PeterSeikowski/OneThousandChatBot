import os

from rag_data_loading.RagWebCrawler import RagWebCrawler
from rag_data_loading.DocumentPreprocessor import DocumentPreprocessor
from rag_vector_store.VectorStore import VectorStore
from rag_chatbot.RagChatBot import RagChatBot



url_name = 'https://onethousand.ai/'

if not os.path.exists("ChromaDBVectorStore"):
    Crawler = RagWebCrawler(url_name, external_urls=['https://www.linkedin.com/'])

    # crawl website for links and content and process to langchain documents
    Crawler.get_links()
    web_content = Crawler.load_web_content()
    document_chunks = DocumentPreprocessor.clean_chunk_transform(web_content)

    #Initialize vector database
    Vector_Store = VectorStore(document_chunks)
    Vector_Store.create_vector_store()

else:
    # initialize vector database
    Vector_Store = VectorStore()
    Vector_Store.create_vector_store()

# Initialize the ChatBot and start a conversation
ChatBot = RagChatBot(Vector_Store)
ChatBot.continual_chat()

