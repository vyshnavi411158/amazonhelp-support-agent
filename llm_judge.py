import os
import json
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


INPUT_FILE = Path("amazonhelp_response_eval_30_to_rate.csv")
OUTPUT_FILE = Path("llm_judge_results.csv")


JUDGE_PROMPT = """
You are evaluating an AI customer-support response.

Score the response from 1 to 5 on each dimension:

1. relevance:
Does the response address the customer's actual issue?

2. helpfulness:
Does it provide a useful next step?

3. grounding:
Is the response consistent with the historical AmazonHelp response
provided, without inventing unsupported facts?

4. appropriateness:
Is the response appropriate for the predicted intent and situation?

Return ONLY valid JSON:

{
  "relevance": 1-5,
  "helpfulness": 1-5,
  "grounding": 1-5,
  "appropriateness": 1-5,
  "overall": 1-5,
  "reason": "brief explanation"
}
"""


def judge_row(client, row):
    user_prompt = f"""
Customer message:
{row['query']}

Predicted intent:
{row['intent']}

Model confidence:
{row['confidence']}

Historical response match score:
{row['historical_similarity']}

System response:
{row['response']}

Decision:
{row['decision']}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return json.loads(response.choices[0].message.content)


def main():
    if OpenAI is None:
        raise RuntimeError(
            "Install the OpenAI package first: pip install openai"
        )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "No LLM-judge results have been generated."
        )

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_FILE)

    results = []

    for _, row in df.iterrows():
        scores = judge_row(client, row)

        results.append({
            **row.to_dict(),
            **scores,
        })

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(output)} LLM-judge evaluations to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()