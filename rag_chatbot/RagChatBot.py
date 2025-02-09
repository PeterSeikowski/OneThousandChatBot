from dotenv import load_dotenv
import os

from rag_vector_store.VectorStore import VectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq



class RagChatBot:
    def __init__(self,
                 vector_db: VectorStore,
                 num_retrievals: int = 3
                 ):
        if not vector_db.vector_store:
            raise ValueError('Vector store does not exist, please create or load vector store.')
        self.vector_store = vector_db.vector_store

        # Load the API key and the model
        load_dotenv()
        self.api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=self.api_key)

        self.retriever = vector_db.vector_store.as_retriever(search_type="similarity",
                                                             search_kwargs={"k": num_retrievals},
                                                            )

        # Contextualize question prompt
        # This system prompt helps the AI understand that it should reformulate the question
        # based on the chat history if needed
        self.contextualize_q_system_prompt = (
            "Given the following conversation and a follow up question, "
            "rephrase the follow up question to be a standalone question."
            "Conversation:"
            )

        # Create a prompt template for contextualizing questions
        self.contextualize_q_prompt = ChatPromptTemplate(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("system", "follow up question:"),
                ("human", "{input}"),
            ]
            )

        # Create a history-aware retriever
        # This uses the LLM to help reformulate the question based on chat history
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.contextualize_q_prompt
            )

        # Answer question prompt
        # This system prompt helps the AI understand that it should provide concise answers
        # based on the retrieved context and indicates what to do if the answer is unknown
        self.qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "\n\n"
            "{context}"
            )

        # Create a prompt template for answering questions
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                ("human", "{input}"),
            ]
            )

        # Create a chain to combine documents for question answering
        # `create_stuff_documents_chain` feeds all retrieved context into the LLM
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        # Create a retrieval chain that combines the history-aware retriever and the question answering chain
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def continual_chat(self) -> None:
        print("Start chatting with the AI! Type 'exit' to end the conversation.")
        chat_history = []  # Collect chat history here (a sequence of messages)
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            # Process the user's query through the retrieval chain
            result = self.rag_chain.invoke({"input": query, "chat_history": chat_history})
            # Display the AI's response
            print(f"AI: {result['answer']}")
            # Update the chat history
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=result["answer"]))


    def get_retrieval(self) -> None:
        """
            A test function that prints the retrieved chunks for the given query


            :return:
            """
        while True:
            query = input("You: ")
            if query.lower() == 'exit':
                break
            retrieved_chunks = self.retriever.invoke(query)
            for chunk in retrieved_chunks:
                print(chunk.page_content)
                print('----------------------------------------')














