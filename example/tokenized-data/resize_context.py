import argparse
import os
from itertools import chain
from typing import Any, Dict

from datasets import load_dataset, load_from_disk

def resize_context(
    input_path: str, 
    output_path: str, 
    new_context_length: int = 4096,
    batch_size: int = 1000,
    num_proc: int = 8
):
    # ==========================
    # 1. Load existing tokenized dataset
    # ==========================
    print(f"Loading tokenized dataset from: {input_path}")
    if input_path.endswith(".jsonl") or input_path.endswith(".json"):
        dataset = load_dataset("json", data_files=input_path, split="train")
    elif input_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=input_path, split="train")
    else:
        # Default to Hugging Face saved disk format
        dataset = load_from_disk(input_path)
        
    print(f"Original dataset size: {len(dataset)} blocks (Current context length)")

    # ==========================
    # 2. Regrouping Logic
    # ==========================
    def regroup_texts(examples: Dict[str, list[Any]]) -> Dict[str, list[Any]]:
        concatenated = {k: list(chain.from_iterable(v)) for k, v in examples.items()}
        
        first_key = list(concatenated.keys())[0]
        total_length = len(concatenated[first_key])
        
        if total_length >= new_context_length:
            total_length = (total_length // new_context_length) * new_context_length

        result = {
            k: [t[i : i + new_context_length] for i in range(0, total_length, new_context_length)]
            for k, t in concatenated.items()
        }
        return result

    # ==========================
    # 3. Apply Resizing
    # ==========================
    print(f"Repacking tokens into new context length: {new_context_length}...")
    resized_dataset = dataset.map(
        regroup_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=f"Resizing to {new_context_length} blocks"
    )

    # ==========================
    # 4. Save to Disk
    # ==========================
    print(f"\nSaving resized dataset to: {output_path}")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if output_path.endswith(".jsonl") or output_path.endswith(".json"):
        resized_dataset.to_json(output_path, force_ascii=False)
    elif output_path.endswith(".parquet"):
        resized_dataset.to_parquet(output_path)
    else:
        resized_dataset.save_to_disk(output_path)
        
    print(f"Process Complete! New dataset size: {len(resized_dataset)} blocks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize tokenized dataset context length (e.g., 8192 -> 4096).")
    
    parser.add_argument("--input_path", type=str, required=True, help="Path to the original tokenized dataset (HF disk format, .jsonl, or .parquet).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the resized dataset.")
    parser.add_argument("--new_context_length", type=int, default=4096, help="The new context length (default: 4096).")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing (default: 1000).")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of CPU cores to use (default: 32).")

    args = parser.parse_args()
    
    resize_context(
        input_path=args.input_path,
        output_path=args.output_path,
        new_context_length=args.new_context_length,
        batch_size=args.batch_size,
        num_proc=args.num_proc
    )