
import time
from tqdm.auto import tqdm
from rag_helper import RAGBase


def calc_price(usage, input_price=0.96, output_price=3.84):
    input_cost = usage.input_tokens / 1_000_000 * input_price
    output_cost = usage.output_tokens / 1_000_000 * output_price

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages, input_price=0.96, output_price=3.84):
    return sum(
        calc_price(u, input_price, output_price)["total_cost"]
        for u in usages
    )


def llm_structured(client, instructions, user_prompt, output_type,
                   model="qwen3.7-max-2026-06-08"):
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": instructions},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.output_text.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()

    try:
        result = output_type.model_validate_json(raw)
    except Exception:
        for verdict in ["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]:
            if verdict in raw.upper():
                result = output_type(
                    relevance=verdict,
                    explanation=raw,
                )
                break
        else:
            raise

    return result, response.usage


def llm_structured_retry(client, instructions, user_prompt, output_type,
                         model="qwen3.7-max-2026-06-08", max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client, instructions, user_prompt, output_type, model
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class CurrencyRAG(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages.clear()
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {
            "code": 3.0,
            "name": 2.0,
            "symbol": 2.0,
            "countries": 2.0,
            "decimal_digits": 0.5,
        }

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
        )

    def llm(self, prompt):
        response = self.llm_client.responses.create(
            model=self.model,
            input=[
                {"role": "developer", "content": self.instructions},
                {"role": "user", "content": prompt},
            ],
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, func):
    with tqdm(total=len(seq)) as progress:
        futures = [
            pool.submit(func, item) for item in seq
        ]

        for future in futures:
            future.add_done_callback(lambda _: progress.update())

        return [future.result() for future in futures]
