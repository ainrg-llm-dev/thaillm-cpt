import argparse
import os
import shutil
from glob import glob
from itertools import chain
from typing import Any, Dict

from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

def preprocess_and_tokenize(
    input_dir: str, 
    output_path: str, 
    model_name: str,
    context_length: int = 2048,
    batch_size: int = 1000,
    num_proc: int = 8,
    num_chunks: int = 5
):
    # ==========================
    # Phase 1: Data Gathering
    # ==========================
    file_list = []
    for file_path in input_dir:
        files = glob(file_path)
        file_list.extend(files)
    
    if not file_list:
        raise ValueError(f"No files found in {input_dir}")

    list_of_ds = []
    print(f"Found {len(file_list)} files. Starting loading process...")
    
    for f in file_list:
        try:
            file_format = "parquet" if f.endswith(".parquet") else "json"
            temp_ds = load_dataset(file_format, data_files=f, split="train")
            
            if "text" in temp_ds.column_names:
                temp_ds = temp_ds.select_columns(["text"])
                list_of_ds.append(temp_ds)
                print(f"Loaded: {os.path.basename(f)}")
            else:
                print(f"Skipped: {os.path.basename(f)} - No 'text' column found.")
                
        except Exception as e:
            print(f"Error loading {os.path.basename(f)}: {e}")

    if not list_of_ds:
        raise ValueError("No valid datasets containing a 'text' column were found.")

    print("\nMerging datasets...")
    raw_dataset = concatenate_datasets(list_of_ds)
    print(f"Total raw examples collected: {len(raw_dataset)}")

    print("Cleaning data (removing nulls and empty texts)...")
    def filter_empty_texts(examples):
        return [
            text is not None and len(str(text).strip()) > 0 
            for text in examples["text"]
        ]
        
    raw_dataset = raw_dataset.filter(
        filter_empty_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc
    )
    print(f"Examples after cleaning: {len(raw_dataset)}")
    
    # ==========================
    # Phase 2 & 3: Tokenize & Group in Chunks
    # ==========================
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    eos_token = tokenizer.eos_token
    if not eos_token:
        raise ValueError("Tokenizer does not have an eos_token configured.")

    def tokenize_function(examples: Dict[str, list[Any]]) -> Dict[str, list[Any]]:
        text_examples = [text + eos_token for text in examples["text"]]
        return tokenizer(text_examples, add_special_tokens=False)

    def group_texts(examples: Dict[str, list[Any]]) -> Dict[str, list[Any]]:
        concatenated_examples = {k: list(chain(*v)) for k, v in examples.items()}
        first_key = list(concatenated_examples.keys())[0]
        total_length = len(concatenated_examples[first_key])
        
        if total_length >= context_length:
            total_length = (total_length // context_length) * context_length
            
        result = {
            k: [t[i : i + context_length] for i in range(0, total_length, context_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    temp_dir = "./temp_chunks"
    os.makedirs(temp_dir, exist_ok=True)
    saved_chunk_paths = [] # เก็บแค่ Path

    print(f"\nSplitting processing into {num_chunks} chunks (20% each) to prevent RAM OOM...")

    for i in range(num_chunks):
        print(f"\n--- Processing Chunk {i+1}/{num_chunks} ---")
        
        chunk_ds = raw_dataset.shard(num_shards=num_chunks, index=i)
        print(f"Chunk size: {len(chunk_ds)} examples")

        tokenized_chunk = chunk_ds.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=chunk_ds.column_names,
            desc=f"Tokenizing Chunk {i+1}",
        )

        lm_chunk = tokenized_chunk.map(
            group_texts,
            batched=True,
            batch_size=batch_size, 
            num_proc=num_proc,
            desc=f"Grouping Chunk {i+1}",
        )

        # Save ลง Disk เป็นไฟล์จริงๆ และเก็บ Path ไว้
        chunk_path = os.path.join(temp_dir, f"chunk_{i}")
        print(f"Saving Chunk {i+1} to disk: {chunk_path}")
        lm_chunk.save_to_disk(chunk_path)
        saved_chunk_paths.append(chunk_path)

        # คืน RAM 
        del chunk_ds
        del tokenized_chunk
        del lm_chunk

    # ==========================
    # Phase 4: Final Merge and Saving
    # ==========================
    print("\nLoading all chunks from disk and merging...")
    loaded_chunks = [load_from_disk(path) for path in saved_chunk_paths]
    final_lm_dataset = concatenate_datasets(loaded_chunks)

    print(f"\nSaving final merged dataset to {output_path}")
    
    if output_path.endswith(".jsonl") or output_path.endswith(".json"):
        final_lm_dataset.to_json(output_path, force_ascii=False)
    elif output_path.endswith(".parquet"):
        final_lm_dataset.to_parquet(output_path)
    else:
        final_lm_dataset.save_to_disk(output_path)
        
    print(f"Process Complete! Final tokenized blocks: {len(final_lm_dataset)}")

    print("Cleaning up temporary chunk files...")
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End JSON/Parquet merger, tokenizer, and packer (Chunked & Save to Disk).")
    
    parser.add_argument("--input_dir", type=str, nargs="+", required=True, help="Directories containing raw JSONL/Parquet files.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path (use .jsonl, .parquet, or directory for HF format).")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face Model ID for the tokenizer.")
    parser.add_argument("--context_length", type=int, default=2048, help="Max sequence length for training.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for map operations.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of CPU cores to use.")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunk. Increase this if OOM")
    args = parser.parse_args()
    
    preprocess_and_tokenize(
        input_dir=args.input_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        context_length=args.context_length,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        num_chunks=args.num_chunks
    )