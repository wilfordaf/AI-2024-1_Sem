import os
import warnings
from typing import List

from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.gigachat import GigaChat
from llama_index import PromptHelper, ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import LangchainEmbedding

warnings.filterwarnings("ignore")


def get_giga(giga_key: str) -> GigaChat:
    """Returns an instance of GigaChat with the provided key."""
    return GigaChat(credentials=giga_key, timeout=30, verify_ssl_certs=False)


def test_giga() -> None:
    load_dotenv(".env")
    giga_key = os.getenv("SB_AUTH_DATA")
    assert get_giga(giga_key)
    print(f"GigaChat instance initialized with key: {giga_key[:4]}***")


def get_prompt(user_content: str) -> list[SystemMessage | HumanMessage]:
    """Builds a prompt consisting of a system message and a human message."""
    system_message = SystemMessage(content="You are a helpful assistant.")
    user_message = HumanMessage(content=user_content)
    return [system_message, user_message]


def test_prompt() -> None:
    load_dotenv(".env")
    giga_key = os.getenv("SB_AUTH_DATA")
    giga = get_giga(giga_key)

    user_content = "Hello!"
    prompt = get_prompt(user_content)

    res = giga.invoke(prompt)
    print(res.content)


def get_prompt_few_shot(number: str) -> list[HumanMessage]:
    examples = [
        HumanMessage(content="What is the number of even digits in 11223344?"),
        HumanMessage(content="Answer: The number 11223344 consist of four even digits."),
        HumanMessage(content="What is the number of even digits in 246810?"),
        HumanMessage(content="Answer: The number 246810 consist of five even digits."),
    ]
    user_message = HumanMessage(content=f"What is the number of even digits in {number}?")
    return examples + [user_message]


def test_few_shot() -> None:
    load_dotenv(".env")
    giga_key = os.getenv("SB_AUTH_DATA")
    giga = get_giga(giga_key)

    number = "62388712774"
    prompt = get_prompt_few_shot(number)

    res = giga.invoke(prompt)
    print(res.content)


class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.path_to_data = path_to_data
        self.llm = llm
        self.system_prompt = SystemMessage(
            content="You are a Q&A assistant. Answer questions accurately based on the provided documents."
        )

        self.index = self.build_index()
        self.query_engine = self.index.as_query_engine()

    def build_index(self) -> VectorStoreIndex:
        documents = SimpleDirectoryReader(self.path_to_data).load_data()

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        context_window_size = 4096
        num_output_tokens = 256

        prompt_helper = PromptHelper(
            context_window=context_window_size,
            num_output=num_output_tokens,
            chunk_overlap_ratio=0.1,
        )

        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=LangchainEmbedding(embedding_model),
            prompt_helper=prompt_helper,
        )

        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        return index

    def retrieve_documents(self, query: str) -> List[str]:
        response = self.query_engine.query(query)
        retrieved_docs = [doc.text for doc in response.source_nodes]
        return retrieved_docs

    def query(self, user_prompt: str) -> str:
        relevant_docs = self.retrieve_documents(user_prompt)

        context = "\n".join(relevant_docs)
        system_message_with_context = SystemMessage(
            content=f"{self.system_prompt.content} Here is some relevant context:\n{context}"
        )

        user_message = HumanMessage(content=user_prompt)
        prompt = [system_message_with_context, user_message]

        response = self.llm.invoke(prompt)
        return response.content


def test_llama_index() -> None:
    load_dotenv(".env")
    giga_key = os.getenv("SB_AUTH_DATA")
    giga_pro = GigaChat(credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)
    llama_index = LlamaIndex("./data/", giga_pro)
    res = llama_index.query("What is 'Attention is all you need'?")
    print(res)
