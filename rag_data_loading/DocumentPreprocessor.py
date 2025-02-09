import re

from unstructured.cleaners.core import clean
from unstructured.documents.elements import Element
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata


class DocumentPreprocessor:
    """
    A class that provides functions to clean a list of unstructured.Element objects
    and further processes them into a list of langchain.Document objects for RAG applications.
    """

    def __init__(self):
        pass

    @staticmethod
    def simple_deduplication(elements: list[Element]) -> list[Element]:
        """
        Filters the list of elements for strict duplicates based on their text content.

        :param elements: List of Element objects.
        :return: Deduplicated list of elements.
        """
        unique_texts = set()
        deduplicated_elements = []
        for element in elements:
            if element.text not in unique_texts:
                unique_texts.add(element.text)
                deduplicated_elements.append(element)
        return deduplicated_elements

    @staticmethod
    def advanced_deduplication(elements: list[Element]) -> list[Element]:
        """
        Placeholder for an advanced deduplication strategy based on embedding similarities.

        :param elements: List of Element objects.
        :return: List of elements after deduplication.
        """
        # TODO: Implement more advanced deduplication strategy based on embedding similarities
        return elements

    @staticmethod
    def clean_elements(elements: list[Element]) -> list[Element]:
        """
        Filters elements with empty text and applies unstructured clean function.

        :param elements: List of Element objects.
        :return: List of cleaned elements.
        """
        cleaned_elements = []
        for el in elements:
            if el.text.strip():
                el.text = clean(el.text)
                cleaned_elements.append(el)
        return cleaned_elements

    @staticmethod
    def filter_elements(elements: list[Element],
                        filter_words: set[str] = None) -> list[Element]:
        """
        Filters elements where the text does not contain any of the specified filter words.

        :param elements: List of Element objects.
        :param filter_words: Set of keywords to filter by. Defaults to {'Title', 'Text', 'List'}.
        :return: Filtered list of elements.
        """
        filter_words = filter_words if filter_words is not None else {'Title', 'Text', 'List'}
        return [el for el in elements if any(keywords in el.category for keywords in filter_words)]

    @staticmethod
    def clean_data(elements: list[Element]) -> list[Element]:
        """
        Cleans unstructured elements by applying deduplication, filtering, and cleaning.

        :param elements: List of Element objects.
        :return: Cleaned list of elements.
        """
        elements = DocumentPreprocessor.filter_elements(elements)
        elements = DocumentPreprocessor.clean_elements(elements)
        elements = DocumentPreprocessor.simple_deduplication(elements)
        return elements

    @staticmethod
    def intelligent_chunking(elements: list[Element],
                             overlap: int = 0,
                             combine_text_under_n_chars: int = 500,
                             max_characters: int = 700) -> list[Element]:
        """
        Chunks text intelligently using the chunk_by_title function from unstructured.

        :param elements: List of Element objects.
        :param overlap: Number of characters to overlap between chunks.
        :param combine_text_under_n_chars: Minimum text length for combining smaller chunks.
        :param max_characters: Maximum character length per chunk.
        :return: List of chunked Element objects.
        """
        chunks = chunk_by_title(elements,
                                overlap=overlap,
                                max_characters=max_characters,
                                combine_text_under_n_chars=combine_text_under_n_chars)
        return chunks

    @staticmethod
    def clean_chunk_text(text: str) -> str:
        """
        Cleans unnecessary line breaks in text chunks.

        :param text: The text to clean.
        :return: Cleaned text.
        """
        cleaned = re.sub(r'\n\s*\n+', '\n', text)
        return cleaned.strip()

    @staticmethod
    def transform_to_document(elements: list[Element]) -> list[Document]:
        """
        Transforms a list of unstructured elements into a list of langchain documents.
        Cleans chunked text and filters metadata.

        :param elements: List of Element objects.
        :return: List of Document objects.
        """
        documents = [Document(page_content=DocumentPreprocessor.clean_chunk_text(el.text),
                              metadata=el.metadata.to_dict())
                     for el in elements]
        documents = filter_complex_metadata(documents)
        return documents

    @staticmethod
    def clean_chunk_transform(elements: list[Element]) -> list[Document]:
        """
        Cleans, chunks, and transforms a list of unstructured elements into langchain documents.

        :param elements: List of Element objects.
        :return: List of processed Document objects.
        """
        elements = DocumentPreprocessor.clean_data(elements)
        chunks = DocumentPreprocessor.intelligent_chunking(elements)
        chunks = DocumentPreprocessor.transform_to_document(chunks)
        return chunks










