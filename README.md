# OneThousandChatBot

A **Retrieval-Augmented Generation (RAG) Chatbot** that leverages web scraping, document processing, and vector-based retrieval to provide intelligent and contextualized responses based on the content of a given website.

## Features
- **Web Crawling**: Extracts internal and external links from a given website.
- **Document Processing**: Cleans, chunks, and transforms extracted text into structured LangChain documents.
- **Vector Database**: Stores documents in a persistent ChromaDB vector store.
- **LLM-Powered Retrieval**: Uses a history-aware retriever to fetch relevant document chunks.
- **Interactive Chatbot**: Answers user queries based on retrieved context, ensuring accuracy and relevance.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.10+
- pip
- Virtual environment (optional but recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/PeterSeikowski/OneThousandChatBot.git
   cd OneThousandChatBot
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your API key for Groq LLM:
     ```env
     GROQ_API_KEY=your-api-key-here
     ```

## Usage
### Running the Chatbot
To start the chatbot, run:
```bash
python main.py
```

### How It Works
1. If no vector database exists, the chatbot:
   - Crawls the specified website.
   - Extracts and processes text.
   - Stores processed documents in a vector database.
2. If a vector database exists, it loads it directly.
3. The chatbot then:
   - Takes user input.
   - Retrieves relevant document chunks.
   - Generates accurate responses using LLM-based inference.
   - 

## Customization
- **Changing the Crawled Website**: Modify `url_name` in `main.py`.
- **Adjusting Retrieval Parameters**: Modify `num_retrievals` in `RagChatBot.py`.
- **Using a different Embedding Model**: Change the model in `VectorStore.py`.
- **Using a different LLM**: Change API key and model in `RagChatBot.py`

## Dependencies
Key dependencies include:
- `BeautifulSoup4` (for web scraping)
- `requests` (for HTTP requests)
- `unstructured` (for text cleaning and chunking)
- `langchain` (for document handling and retrieval)
- `chromadb` (for vector storage)
- `groq` (for LLM inference)

Install them using:
```bash
pip install -r requirements.txt
```



