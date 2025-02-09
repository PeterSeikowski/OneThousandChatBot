import os

from rag_data_loading.RagWebCrawler import RagWebCrawler
from rag_data_loading.DocumentPreprocessor import DocumentPreprocessor
from rag_vector_store.VectorStore import VectorStore
from rag_chatbot.RagChatBot import RagChatBot




# Define the base URL for crawling
url_name = 'https://onethousand.ai/'

# Check if the vector store directory exists
if not os.path.exists("ChromaDBVectorStore"):
    # Initialize the web crawler with the base URL and an external link
    Crawler = RagWebCrawler(url_name, external_urls=['https://www.linkedin.com/'])

    # Crawl the website to gather internal and external links
    Crawler.get_links()

    # Load content from the discovered links
    web_content = Crawler.load_web_content()

    # Process the extracted web content: cleaning, chunking, and transforming into LangChain documents
    document_chunks = DocumentPreprocessor.clean_chunk_transform(web_content)

    # Initialize the vector database with processed documents
    Vector_Store = VectorStore(document_chunks)
    Vector_Store.create_vector_store()

else:
    # Load the existing vector store
    Vector_Store = VectorStore()
    Vector_Store.create_vector_store()

# Initialize the chatbot with the vector store and start an interactive conversation
ChatBot = RagChatBot(Vector_Store)
ChatBot.continual_chat()

