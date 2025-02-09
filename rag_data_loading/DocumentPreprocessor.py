import re

from unstructured.cleaners.core import clean
from unstructured.documents.elements import Element
from unstructured.chunking.title import chunk_by_title
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata



class DocumentPreprocessor:
    """
         A class that provides functions to clean a list of unstructured.Element objects
         and further processes them to a list of langchain.Document objects for RAG applications
    """
    def __init__(self):
        pass

    @staticmethod
    def simple_deduplication(elements: list[Element]) -> list[Element]:
        """
            Filters the list of elements for strict duplicates
        :param list elements:
        :return: deduplicated list of elements
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
        # TODO: Implement more advanced deduplication strategy based on embedding similarities
        return elements

    @staticmethod
    def clean_elements(elements: list[Element]) -> list[Element]:
        """
            Filter elements with empty text and apply unstructured clean function
        :param list of elements:
        :return: list of cleaned elements
        """
        cleaned_elements = []
        for el in elements:
            if el.text.strip():
                el.text = clean(el.text)
                cleaned_elements.append(el)
        return cleaned_elements

    @staticmethod
    def filter_elements(elements: list[Element],
                        filter_words: set[str] = None
                        ) -> list[Element]:
        """
            Filters elements for which the text does not contain any of the filter words
        :param list of elements:
        :param filter_words:
        :return: filtered list of elements
        """
        filter_words = filter_words if filter_words is not None else {'Title', 'Text', 'List'}
        return [el for el in elements if any(keywords in el.category for keywords in filter_words)]

    @staticmethod
    def clean_data(elements: list[Element]) -> list[Element]:
        """
            clean unstructured elements list, i.e. deduplication, filtering, cleaning
        :param list of elements:
        :return: cleaned list of elements
        """
        elements = DocumentPreprocessor.filter_elements(elements)
        elements = DocumentPreprocessor.clean_elements(elements)
        elements = DocumentPreprocessor.simple_deduplication(elements)
        return elements

    @staticmethod
    def intelligent_chunking(elements: list[Element],
                             overlap: int = 0,
                             combine_text_under_n_chars: int = 500,
                             max_characters: int = 700
                            )-> list[Element]:
        """
            Using the chunk_by_title function from unstructured
        :param elements:
        :param overlap:
        :param combine_text_under_n_chars:
        :param max_characters:
        :return:
        """
        chunks = chunk_by_title(elements,
                                overlap=overlap,
                                max_characters=max_characters,
                                combine_text_under_n_chars=combine_text_under_n_chars
                                )
        return chunks

    @staticmethod
    def clean_chunk_text(text: str) -> str:
        """
            Clean unnecessary line breaks in the chunks
        :param text:
        :return:
        """
        cleaned = re.sub(r'\n\s*\n+', '\n', text)
        return cleaned.strip()

    @staticmethod
    def transform_to_document(elements: list[Element]) -> list[Document]:
        """
            Transforms list of unstructured elements to list of langchain documents.
            Filters the metadata of the documents.
            Cleans chunks of unnecessary line breaks.
        :param elements:
        :return:
        """
        documents = [Document(page_content=DocumentPreprocessor.clean_chunk_text(el.text),
                              metadata=el.metadata.to_dict()
                              ) for el in elements]
        documents = filter_complex_metadata(documents)
        return documents

    @staticmethod
    def clean_chunk_transform(elements: list[Element]) -> list[Document]:
        """
            Clean and chunk a list of unstructured elements and transforms it
            to a list of langchain documents.
        :param elements:
        :return:
        """
        elements = DocumentPreprocessor.clean_data(elements)
        chunks = DocumentPreprocessor.intelligent_chunking(elements)
        chunks = DocumentPreprocessor.transform_to_document(chunks)
        return chunks













