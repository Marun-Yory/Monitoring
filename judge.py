
import os

from typing import Literal
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from evaluation_utils import llm_structured_retry


class RelevanceVerdict(BaseModel):
    relevance: Literal[
        "NON_RELEVANT",
        "PARTLY_RELEVANT",
        "RELEVANT",
    ]
    explanation: str


judge_instructions = """
You are an expert evaluator for a Currency Assistant RAG system.

Your task is to evaluate whether the generated answer is relevant
to the user's currency-related question.

Consider the following information when evaluating relevance:

- Currency codes (e.g., USD, EUR, GBP, JPY)
- Currency names (e.g., US Dollar, Euro, British Pound)
- Currency symbols (e.g., $, €, £, ¥)
- Countries associated with currencies
- Currency decimal digits
- General currency-related information

You must respond ONLY with a valid JSON object in this exact schema:

{
  "relevance": "RELEVANT",
  "explanation": "Short explanation of the verdict"
}

The "relevance" value must be exactly one of:

- "RELEVANT": The answer directly and correctly addresses the question.
- "PARTLY_RELEVANT": The answer addresses some aspects of the question
  but is incomplete or misses important information.
- "NON_RELEVANT": The answer does not address the question or provides
  unrelated information.

Evaluate the answer based on its relevance to the question.
Do not require information that was not requested.

Do not include markdown code block formatting.
Return raw JSON only.
""".strip()


judge_prompt = """
Question: {question}

Generated Answer: {answer}
""".strip()


DEFAULT_BASE_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

DEFAULT_MODEL = "qwen3.7-max-2026-06-08"


def evaluate_relevance(
    question,
    answer,
    client=None,
    model=None,
    api_key=None,
    base_url=None,
):
    if client is None:
        load_dotenv()

        api_key = api_key or os.getenv("OPENAI_API_KEY")

        base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        )

        if not api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY "
                "in your environment or .env file."
            )

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    model = (
        model
        or os.getenv("LLM_MODEL", DEFAULT_MODEL)
    )

    prompt = judge_prompt.format(
        question=question,
        answer=answer,
    )

    result, usage = llm_structured_retry(
        client=client,
        instructions=judge_instructions,
        user_prompt=prompt,
        output_type=RelevanceVerdict,
        model=model,
    )

    return result.relevance, result.explanation


if __name__ == "__main__":
    load_dotenv()

    question = "What is the currency used in Japan?"

    answer = (
        "Japan uses the Japanese Yen (JPY), "
        "represented by the symbol ¥. "
        "It has 0 decimal digits."
    )

    relevance, explanation = evaluate_relevance(
        question,
        answer,
    )

    print("Relevance:", relevance)
    print("Explanation:", explanation)
