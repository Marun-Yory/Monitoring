
import time
import random

from metrics import LLMCallRecord
from db_save import save_conversation
from db_feedback import save_feedback


SAMPLE_QUESTIONS = [
    "What currency is used in the United States?",
    "What is the currency code for Japan?",
    "What symbol represents the Euro?",
    "Which countries use the Euro?",
    "How many decimal digits does the Japanese Yen have?",
]

SAMPLE_ANSWERS = [
    "The United States uses the US Dollar (USD), represented by $.",
    "Japan uses the Japanese Yen (JPY), represented by ¥.",
    "The Euro currency symbol is €.",
    "Germany, France, Italy, Spain, and the Netherlands use the Euro.",
    "The Japanese Yen has 0 decimal digits.",
]

RELEVANCE = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]


def fake_record(question, answer):
    prompt_tokens = random.randint(50, 200)
    completion_tokens = random.randint(50, 300)

    return LLMCallRecord(
        model="qwen3.7-max-2026-06-08",
        prompt=question,
        instructions="",
        answer=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        response_time=random.uniform(0.5, 5.0),
        cost=random.uniform(0.0001, 0.01),
    )


def random_score():
    return random.choice([1, 1, 1, 1, -1])


def generate_one():
    question = random.choice(SAMPLE_QUESTIONS)
    answer = random.choice(SAMPLE_ANSWERS)

    record = fake_record(question, answer)

    conversation_id = save_conversation(record, question)

    if random.random() < 0.7:
        relevance = random.choice(RELEVANCE)

        save_feedback(
            conversation_id,
            "judge",
            relevance=relevance,
            explanation=f"Answer is {relevance.lower()}.",
        )

    if random.random() < 0.5:
        save_feedback(
            conversation_id,
            "user",
            score=random_score(),
        )


def generate_live():
    print("Starting live data generation (Ctrl+C to stop)...", flush=True)

    while True:
        generate_one()
        time.sleep(1)


if __name__ == "__main__":
    try:
        generate_live()
    except KeyboardInterrupt:
        print("Stopped.")
