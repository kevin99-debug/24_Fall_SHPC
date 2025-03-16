import argparse
import os

from transformers import GPT2Config, GPT2LMHeadModel
from transformers import AutoTokenizer
from datasets import load_dataset

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import torch

#for time
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # training settings
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--global_batch", default=512, type=int, help="Total batch size across all GPUs and gradient accumulation steps")
    parser.add_argument("--micro_batch", default=2, type=int, help="Batch size processed per GPU per step (before gradient accumulation)")
    parser.add_argument("--epochs", default=3, type=int)
    
    # ZeRO
    parser.add_argument("--stage", default=0, type=int)
    parser.add_argument("--offload", action="store_true", help="Enable CPU/NVMe offloading")
    parser.add_argument("--dtype", default="fp32", type=str, choices=["fp32", "fp16"], help="Precision for training")
    args = parser.parse_args()
    
    # Deepspeed optimizer
    adam_optimizer = {
        "type": "Adam",
        "params": {
            "lr": args.learning_rate,
        }
    }
        
    # ZeRO Stage 0: No ZeRO optimization (baseline, DDP)
    stage_0_config = {
        "train_batch_size": args.global_batch,
        "train_micro_batch_size_per_gpu":args.micro_batch,
        "zero_optimization": {
            "stage": 0
        },
        "optimizer": adam_optimizer,
    }

    # ZeRO Stage 1: Optimizer State Partitioning
    stage_1_config = {
        "train_batch_size": args.global_batch,
        "train_micro_batch_size_per_gpu":args.micro_batch,
        "zero_optimization": {
            "stage": 1,
            "reduce_scatter": True,
            "allgather_partitions": True
        },
        "optimizer": adam_optimizer,
    }

    # ZeRO Stage 2: Optimizer State, Gradient Partitioning
    stage_2_config = {
        "train_batch_size": args.global_batch,
        "train_micro_batch_size_per_gpu":args.micro_batch,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "overlap_comm": True,
        },
        "optimizer": adam_optimizer,
    }

    # ZeRO Stage 3: Optimizer State, Gradient, Parameter Partitioning + Offloading to CPU
    stage_3_config = {
        "train_batch_size": 64,
        "zero_optimization": {
            "stage": 3,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "sub_group_size": 1e9,
        },
        "optimizer": adam_optimizer,
    }

    #add codes for offloading and FP16
    #offloading
    if args.offload:
        if args.stage == 2:
            stage_2_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }
            stage_2_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True
            }
        if args.stage == 3:
            stage_3_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }
            stage_3_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True
            }
    #FP16
    if args.stage == 3 and args.dtype == "fp16":
        fp16_config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 32,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
        }
        stage_3_config.update(fp16_config)


    if args.stage == 0:
        deepspeed_config = stage_0_config
    elif args.stage == 1:
        deepspeed_config = stage_1_config
    elif args.stage == 2:
        deepspeed_config = stage_2_config
    elif args.stage == 3:
        deepspeed_config = stage_3_config

    
    # model & tokenizer
    model_name = "gpt2-large"
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # dataset & dataloader
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_dataloader = torch.utils.data.DataLoader(tokenized_datasets, 
                                                   batch_size=args.micro_batch, 
                                                   shuffle=True)
    
    # DeepSpeed init
    deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=deepspeed_config
    )

    # Slurm Process ID
    slurm_id = int(os.environ["SLURM_PROCID"])
    slurm_tasks_per_node = int(os.environ["GPUS_PER_NODE"])
    global_rank = slurm_id * slurm_tasks_per_node + local_rank
    
    # training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            if step == 20:
                break
            # empty cache
            get_accelerator().empty_cache()

            # prepare batch
            inputs = {key: val.to(model_engine.local_rank) for key, val in batch.items()}
            labels = inputs["input_ids"]

            # forward
            outputs = model_engine(**inputs, labels=labels)
            loss = outputs.loss
            #memory check after FORWARD!!!
            mem_forward = torch.cuda.memory_allocated(device=model_engine.local_rank)
            if global_rank == 0 and step % 10 == 0:
                print(f"[Forward] Step {step}, GPU Mem Used: {mem_forward} bytes")

            # backward
            model_engine.backward(loss)
            #memory check after BACKWARD!!!
            mem_backward = torch.cuda.memory_allocated(device=model_engine.local_rank)
            if global_rank == 0 and step % 10 == 0:
                print(f"[Backward] Step {step}, GPU Mem Used: {mem_backward} bytes")
           
            # optimizer step 
            model_engine.step()
            mem_optimizer = torch.cuda.memory_allocated(device=model_engine.local_rank)
            if global_rank == 0 and step % 10 == 0:
                print(f"[Optimizer Step] Step {step}, GPU Mem Used: {mem_optimizer} bytes")


            # print status
            if step % 10 == 0 and global_rank == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")
        #time
        end_time = time.time()
        elapsed = end_time - start_time
        if global_rank == 0:
            print(f"20 iterations took {elapsed} seconds.")

    print("Training complete.")
