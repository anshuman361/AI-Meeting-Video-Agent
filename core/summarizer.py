import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found.\n"
        )

    return ChatMistralAI(
        model="ministral-8b-latest",      
        mistral_api_key=api_key,
        temperature=0.2,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=100,
    )
    return splitter.split_text(transcript)


# --------------------------------------------------
# Summarizer
# --------------------------------------------------
def summarize(transcript: str) -> str:
    llm = get_llm()

    # Small transcript → Single API call
    if len(transcript) <= 8000:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Summarize the meeting into concise professional bullet points.",
                ),
                ("human", "{text}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"text": transcript})

    # Large transcript → Map Reduce
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of the meeting transcript briefly in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # Parallel API Calls
    chunk_inputs = [{"text": chunk} for chunk in chunks]
    chunk_summaries = map_chain.batch(chunk_inputs)

    combined_summary = "\n\n".join(chunk_summaries)

    combine_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Combine these summaries into one professional meeting summary."
                "Return concise bullet points without repetition.",
            ),
            ("human", "{text}"),
        ]
    )

    combine_chain = combine_prompt | llm | StrOutputParser()

    return combine_chain.invoke({"text": combined_summary})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a short professional meeting title (maximum 8 words). "
                "Return only the title.",
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"text": transcript[:1000]})