# you can download the CSV from https://wandb.ai/costa-huang/tldr_summarize/runs/gb2dian5

import asyncio
import random
from dataclasses import dataclass
import time
from typing import Optional

import pandas as pd
import tyro
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio



@dataclass
class Args:
    csv: str = "trained_response.csv"
    output_path: Optional[str] = None
    n: int = 64
    num_trails: int = 1
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: Optional[int] = None

    def __post_init__(self):
        if "gpt-3.5" in self.model:
            # gpt-3.5 generates so fast that it will exceeds the
            # token limit per minute
            self.max_parallel_requests = 11
        elif "gpt-4" in self.model:
            self.max_parallel_requests = 13
    
args = tyro.cli(Args)
if args.output_path is None:
    args.output_path = args.csv.split(".csv")[0] + f"{args.model}.csv"

limiter = asyncio.Semaphore(args.max_parallel_requests)
async_client = AsyncOpenAI()

template = r"""
Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? Judge based on accuracy, coverage, and coherence.

### Post:
{{post}}

### Summary A:
{{summarya}}

### Summary B:
{{summaryb}}

### Instructions:
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
"""


async def process_text(post, summary_a, summary_b, i):
    text = template.replace("{{post}}", post)
    text = text.replace("{{summarya}}", summary_a)
    text = text.replace("{{summaryb}}", summary_b)  # Ensure this split logic is correct for your data

    async with limiter:
        response = None
        while response is None:
            try:
                response = await async_client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": text},
                    ],
                )
                r = response.choices[0].message.content
            except Exception as e:
                print(f"error in {i}: {e}")
                time.sleep(30)
                continue
        
        # print()
        try:
            comparison = r.split("Comparison:")[1].split("Preferred:")[0].strip()
            preferred = r.split("Preferred:")[1].strip()
            return comparison, preferred, i, text + r
        except:
            print(f"error in {i}")
            return "", random.choice(["A", "B"]), i, text + r


async def main(args: Args):
    for j in range(args.num_trails):
        print(f"trail {j}")
        tasks = []
        df = pd.read_csv(args.csv)
        df["explanation"] = [None for _ in range(len(df))]
        df["preferred"] = [None for _ in range(len(df))]
        df["shuffled_index"] = [None for _ in range(len(df))]
        df["entire_conversation"] = [None for _ in range(len(df))]
        r = range(min(args.n, len(df)))
        if args.n == -1:
            r = range(len(df))
        for i in r:
            post = df["query"].iloc[i].strip()
            # shuffled the index to avoid GPT4's preference bias in the content's order
            shuffled_index = random.randint(0, 1)
            df.at[i, "shuffled_index"] = shuffled_index
            summaries = [
                df["postprocessed_response"].iloc[i].strip(),
                df["reference_responses"].iloc[i].split("<|endoftext|>")[0].strip(),
            ]
            summary_a = summaries[shuffled_index]
            summary_b = summaries[1 - shuffled_index]
            task = asyncio.create_task(process_text(post, summary_a, summary_b, i))
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks)

        for _, (comparison, preferred, i, entire_conversation) in enumerate(results):
            df.at[i, "explanation"] = comparison
            df.at[i, "entire_conversation"] = entire_conversation
            preferred_label = (
                "ours"
                if (df.at[i, "shuffled_index"] == 0 and preferred == "A")
                or (df.at[i, "shuffled_index"] == 1 and preferred == "B")
                else "reference"
            )
            df.at[i, "preferred"] = preferred_label

        print(df["preferred"].value_counts())
        df.to_csv(args.output_path)
        # return df


asyncio.run(main(args))
