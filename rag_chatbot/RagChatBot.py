from dotenv import load_dotenv
import os

from rag_vector_store.VectorStore import VectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq



class RagChatBot:
    """
    A Retrieval-Augmented Generation (RAG) chatbot using a vector database for knowledge retrieval
    and an LLM for generating responses. This chatbot supports context-aware question reformulation
    and concise answer generation.

    Attributes:
        vector_store (Chroma): The vector database storing embeddings of documents.
        api_key (str): API key for accessing the Groq LLM service.
        llm (ChatGroq): The language model used for generating responses.
        retriever (Retriever): Retrieves relevant document chunks from the vector store.
        contextualize_q_prompt (ChatPromptTemplate): Reformulates user questions to be self-contained.
        history_aware_retriever (Retriever): Enhances retrieval by taking chat history into account.
        qa_prompt (ChatPromptTemplate): The prompt template for generating concise answers.
        question_answer_chain (Chain): Chain combining retrieved documents into a response.
        rag_chain (Chain): The full RAG pipeline including retrieval and answer generation.
    """

    def __init__(self, vector_db: VectorStore, num_retrievals: int = 3):
        """
        Initializes the RAG chatbot by setting up the vector store, retrieval mechanisms,
        and LLM-based response generation.

        Args:
            vector_db (VectorStore): The vector store containing embedded documents.
            num_retrievals (int): Number of document chunks to retrieve per query.
        """
        if not vector_db.vector_store:
            raise ValueError('Vector store does not exist, please create or load vector store.')
        self.vector_store = vector_db.vector_store

        # Load API key and LLM model
        load_dotenv()
        self.api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=self.api_key)

        # Define the retriever
        self.retriever = vector_db.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": num_retrievals})

        # Contextualize question prompt
        self.contextualize_q_system_prompt = (
            "Given the following conversation and a follow-up question, "
            "rephrase the follow-up question to be a standalone question."
            "Conversation:"
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", self.contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("system", "Follow-up question:"),
            ("human", "{input}"),
        ])

        # History-aware retriever
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
        )

        # Answer generation prompt
        self.qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "\n\n"
            "{context}"
        )

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.qa_system_prompt),
            ("human", "{input}"),
        ])

        # Create answer generation chain
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        # Create the full retrieval-augmented generation (RAG) pipeline
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def continual_chat(self) -> None:
        """
        Starts an interactive chat session with the AI. The chatbot maintains chat history
        and uses retrieval-augmented generation to provide responses.

        Type 'exit' to end the conversation.
        """
        print("Start chatting with the AI! Type 'exit' to end the conversation.")
        chat_history = []
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            result = self.rag_chain.invoke({"input": query, "chat_history": chat_history})
            print(f"AI: {result['answer']}")
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=result["answer"]))

    def get_retrieval(self) -> None:
        """
        A test function that prints the retrieved document chunks for a given query.

        Type 'exit' to stop retrieval testing.
        """
        while True:
            query = input("You: ")
            if query.lower() == 'exit':
                break
            retrieved_chunks = self.retriever.invoke(query)
            for chunk in retrieved_chunks:
                print(chunk.page_content)
                print('----------------------------------------')





