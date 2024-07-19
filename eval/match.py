from collections import defaultdict
import pandas as pd
from rich.table import Table
from rich.console import Console

def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)
# Load the data
oai_df = pd.read_csv("sampled_data/openai_ppo_original_1.3b/validation.csv_judged.csv")
our_df = pd.read_csv("sampled_data/refactor-chosen-rejected3/ppo_left_padding_new/EleutherAI/pythia-2.8b-deduped/44413/query_responses_gpt-3.5-turbo-0125.csv")

# Create a dictionary from oai_df for quick lookup
# Use the first 200 characters of the "query" column as the key and the row index as the value
idxs = {oai_df.iloc[i]["query"][:200]: i for i in range(len(oai_df))}

# Use the apply method to vectorize the lookup and assignment operation
# This avoids the loop and directly assigns the matching index to the new 'new_idx' column
# The lambda function looks up each truncated query in the 'idxs' dictionary
# It returns -1 if the truncated query is not found, you can adjust this value based on your needs
our_df['new_idx'] = our_df['query'].apply(lambda x: idxs.get(x[:200], -1))


console = Console()
for i in range(len(oai_df)):
    oai_query = oai_df.iloc[i]["query"]
    # our_query = our_df.iloc[i]["query"]
    our = our_df.loc[our_df['new_idx'] == i].iloc[0]
    our_query = our["query"]

    if oai_query != our_query:
        print(f"Query mismatch at index {i}:\nOAI: {oai_query}\nOur: {our_query}\n")
    
    oai_response = oai_df.iloc[i]["postprocessed_response"]
    our_response = our["postprocessed_response"]

    table = defaultdict(list)
    table["Query"].append(oai_query)
    table["OAI Response"].append(oai_response)
    table["Our Response"].append(our_response)
    print_rich_table(f"Query {i}", pd.DataFrame(table), console)
    input("Press Enter to continue...")
    
