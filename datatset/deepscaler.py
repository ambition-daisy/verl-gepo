from datasets import load_dataset


def make_map_fn(split):
    def process_fn(example,idx):
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = example.pop("problem")
        question = question + " " + instruction
        answer = example.pop("answer")
        data_source = "deepscaler"
        solution = example.pop("solution")
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "index": idx},
        }
        return data

    return process_fn

train_dataset = load_dataset("json",data_files=["/Users/daisy/Desktop/gepo/deepscaler.json"])["train"]
train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

print(train_dataset[0])
train_dataset.to_parquet("/Users/daisy/Desktop/gepo/deepscaler.parquet")