# coding: utf-8
import os
import sys
import re
import requests
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from safetensors import safe_open
import gc

# Cell 1: Installation (Commented out)
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth vllm
# else:
#     # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
#     !pip install --no-deps unsloth vllm

# Cell 2: Colab Extra Install (Commented out)
# #@title Colab Extra Install { display-mode: "form" }
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth vllm
# else:
#     !pip install --no-deps unsloth vllm
#     # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
#     # Skip restarting message in Colab
#     import sys, re, requests; modules = list(sys.modules.keys())
#     for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
#
#     # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
#     f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
#     with open("vllm_requirements.txt", "wb") as file:
#         file.write(re.sub(rb"(transformers|numpy|xformers)[^\\n]{1,}\\n", b"", f))
#     !pip install -r vllm_requirements.txt

# Cell 3: Model and Tokenizer Setup
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

# Cell 4: Define Prompts
reasoning_start = "<start_working_out>" # Acts as <think>
reasoning_end   = "<end_working_out>"   # Acts as </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
# print(system_prompt) # Notebook specific display

# Cell 5: Chat Template Setup
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}{% endif %}"

# Replace with out specific template:
chat_template = chat_template\
    .replace("'{system_prompt}'",   f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

# Cell 6: Apply Chat Template Example (Commented out display)
# tokenizer.apply_chat_template([
#     {"role" : "user", "content" : "What is 1+1?"},
#     {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
#     {"role" : "user", "content" : "What is 2+2?"},
# ], tokenize = False, add_generation_prompt = True)

# Cell 7: Load and Process SFT Dataset (OpenMathReasoning-mini)
# This section is for the SFT part, GRPO dataset loading is later
# from datasets import load_dataset # Already imported
# import pandas as pd # Already imported
# import numpy as np # Already imported
sft_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
sft_dataset = sft_dataset.to_pandas()[
    ["expected_answer", "problem", "generated_solution"]
]
is_number = pd.to_numeric(pd.Series(sft_dataset["expected_answer"]), errors = "coerce").notnull()
sft_dataset = sft_dataset.iloc[np.where(is_number)[0]]
# sft_dataset # Notebook specific display

# Cell 8: Format SFT Dataset
def format_sft_dataset(x): # Renamed to avoid conflict if script combines SFT and GRPO logic differently
    expected_answer = x["expected_answer"]
    problem = x["problem"]
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")
    thoughts = thoughts.strip()
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + str(expected_answer) + solution_end # Ensure expected_answer is string
    return [
        {"role" : "system",    "content" : system_prompt},
        {"role" : "user",      "content" : problem},
        {"role" : "assistant", "content" : final_prompt},
    ]
sft_dataset["Messages"] = sft_dataset.apply(format_sft_dataset, axis = 1)

# Cell 9: Apply Chat Template to SFT data (Commented out display)
# tokenizer.apply_chat_template(sft_dataset["Messages"][0], tokenize = False)

# Cell 10: Filter SFT Dataset by Sequence Length
sft_dataset["N"] = sft_dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
sft_dataset = sft_dataset.loc[sft_dataset["N"] <= max_seq_length/2].copy()
# sft_dataset.shape # Notebook specific display

# Cell 11: Convert SFT Pandas to Datasets.Dataset
# from datasets import Dataset # Already imported
sft_dataset["text"] = tokenizer.apply_chat_template(sft_dataset["Messages"].values.tolist(), tokenize = False)
sft_dataset = Dataset.from_pandas(sft_dataset)
# sft_dataset # Notebook specific display

# Cell 12: SFT Trainer Setup
# from trl import SFTTrainer, SFTConfig # Already imported
sft_trainer = SFTTrainer( # Renamed trainer to sft_trainer
    model = model,
    tokenizer = tokenizer,
    train_dataset = sft_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)

# Cell 13: SFT Training
# sft_trainer.train() # Commented out for script conversion, can be uncommented to run SFT

# Cell 14: SFT Model Generation Example (Commented out display)
# text_sft_example = tokenizer.apply_chat_template(
#     sft_dataset[0]["Messages"][:2],
#     tokenize = False,
#     add_generation_prompt = True,
# )
# from transformers import TextStreamer # Already imported
# _ = model.generate(
#     **tokenizer(text_sft_example, return_tensors = "pt").to("cuda"),
#     temperature = 0,
#     max_new_tokens = 1024,
#     streamer = TextStreamer(tokenizer, skip_prompt = False),
# )

# Cell 15: Clean up SFT data
del sft_dataset
torch.cuda.empty_cache()
# import gc # Already imported
gc.collect()

# Cell 16: Load GRPO Dataset (DAPO-Math-17k-Processed)
# from datasets import load_dataset # Already imported
# Load the local jsonl dataset
dataset = load_dataset("json", data_files="/home/data/data.jsonl", split="train")
# dataset # Notebook specific display

# Cell 17: Display GRPO Dataset Prompt Example (Commented out)
# dataset[0]["prompt"]

# Cell 18: Display GRPO Dataset Solution Example (Commented out)
# dataset[0]["solution"]

# Cell 19: Define extract_hash_answer
def extract_hash_answer(text):
    if "####" in text:
        return text.split("####")[1].strip()
    # Return the original text if "####" is not found,
    # or handle as an error, though returning original might be safer
    # depending on downstream use if the new dataset isn't perfectly clean.
    # For now, let's return None if "####" is not present, aligning with the original commented code's intent.
    return None
# The line extract_hash_answer(dataset[0]["solution"]) is for testing and might error
# if dataset is not defined at this point in the notebook flow when converting to .py.
# For now, keep it as is, but be mindful if it causes issues during .py conversion.
# print(extract_hash_answer(dataset[0]["solution"])) # Example usage

# Cell 20: Map GRPO Dataset
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["prompt"]},
    ],
    "answer": extract_hash_answer(x["solution"]),
})
# dataset[0] # Notebook specific display

# Cell 21: Regex for Solution Matching
# import re # Already imported
solution_end_regex = r"</SOLUTION>[\\s]{0,}" + \
    "(?:" + re.escape(tokenizer.eos_token) + ")?"
match_format = re.compile(
    rf"{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end_regex}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)
# match_format # Notebook specific display

# Cell 22: Regex Match Example 1 (Commented out display)
# match_format.findall(
#     "Let me think!<end_working_out>"\
#     f"<SOLUTION>\\n2\\n</SOLUTION>",
# )

# Cell 23: Regex Match Example 2 (Commented out display)
# match_format.findall(
#     "<start_working_out>Let me think!<end_working_out>"\
#     f"<SOLUTION>  2  </SOLUTION>\\n\\n",
# )

# Cell 24: Define match_format_exactly Reward Function
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

# Cell 25: Define match_format_approximately Reward Function
def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(solution_start)  == 1 else -1.0
        score += 0.5 if response.count(solution_end)    == 1 else -1.0
        scores.append(score)
    return scores

# Cell 26: Define check_answer Reward Function
def check_answer(prompts, completions, answer, **kwargs):
    # question = prompts[0][-1]["content"] # Not used in current logic
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        if true_answer is None: # Handle cases where true_answer might be None
            scores.append(-2.0) # Or some other appropriate score
            continue
        if guess == true_answer:
            score += 5.0
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 1.5
                else: score -= 2.5
            except:
                score -= 4.5
        scores.append(score)
    return scores

# Cell 27: Regex for Number Matching and Examples (Commented out prints)
match_numbers = re.compile(
    solution_start + r".*?[\\s]{0,}([-]?[\\d\\.\\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
# print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
# print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
# print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
# print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))

# Cell 28: Define check_numbers Reward Function
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]
    scores = []
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        # Ensure answer[0] is not None before trying to print or use it
        true_answer_display = answer[0] if answer and answer[0] is not None else "N/A"
        print(
            '*'*20 + f"Question:\\n{question}", f"\\nAnswer:\\n{true_answer_display}", f"\\nResponse:\\n{responses[0]}", f"\\nExtracted:\\n{extracted_responses[0]}"
        )
    PRINTED_TIMES += 1

    for guess, true_answer_item in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        if true_answer_item is None: # Check if true_answer_item itself is None
             scores.append(-2.5) # Or other score if no true answer to compare against
             continue
        try:
            true_answer_val = float(true_answer_item.strip())
            guess_val       = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess_val == true_answer_val else -1.5)
        except:
            scores.append(0)
            continue
    return scores

# Cell 29: Tokenize and Filter GRPO Dataset
tokenized = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
)
# print(tokenizer.decode(tokenized[0]["tokens"])) # Commented out display
tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

# import numpy as np # Already imported
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized
gc.collect()

# Cell 30: GRPO Trainer Setup (Train the model section)
max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

# from vllm import SamplingParams # Already imported
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

# from trl import GRPOConfig, GRPOTrainer # Already imported
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1, # Will be adjusted by GRPOTrainer if num_generations > 1
    gradient_accumulation_steps = 1,
    num_generations = 4,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 100,
    save_steps = 100,
    report_to = "none",
    output_dir = "outputs",
)

# Cell 31: GRPO Trainer Initialization and Training
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer, # Pass tokenizer for processing
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# Cell 32: GRPO Model Generation Example (Commented out display)
# text_grpo_example = "What is the sqrt of 101?" # Simple text, not from dataset messages
# from vllm import SamplingParams # Already imported
# sampling_params_grpo = SamplingParams(
#     temperature = 1.0,
#     top_k = 50,
#     max_tokens = 1024,
# )
# output_grpo = model.fast_generate(
#     [text_grpo_example], # Needs to be a list
#     sampling_params = sampling_params_grpo,
#     lora_request = None, # Assuming we test base model or merged
# )[0].outputs[0].text
# print(output_grpo)

# Cell 33: Save LoRA Adapters
model.save_lora("grpo_saved_lora")

# Cell 34: Verify Saved Tensors
# from safetensors import safe_open # Already imported
tensors = {}
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum().item() # .item() to get Python number
        # Ensure not all elements are zero, but allow some zeros.
        # Original assert was (tensor == 0).sum() / tensor.numel() != tensor.numel() which is always true if not all are zero.
        # A more meaningful check is if the tensor is not entirely zero.
        assert(n_zeros != tensor.numel())
print("LoRA tensors verified.")

# Cell 35: GRPO Model Generation with Loaded LoRA (Commented out display)
# messages_grpo_lora = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user",   "content": "What is the sqrt of 101?"},
# ]
# text_grpo_lora = tokenizer.apply_chat_template(
#     messages_grpo_lora,
#     add_generation_prompt = True,
#     tokenize = False,
# )
# from vllm import SamplingParams # Already imported
# sampling_params_grpo_lora = SamplingParams(
#     temperature = 1.0,
#     top_k = 50,
#     max_tokens = 2048,
# )
# output_grpo_lora = model.fast_generate(
#     text_grpo_lora, # fast_generate expects a string or list of strings
#     sampling_params = sampling_params_grpo_lora,
#     lora_request = model.load_lora("grpo_saved_lora"),
# )[0].outputs[0].text
# print(output_grpo_lora)

# Cell 36 & 37: Model Saving Options (Commented out)
# These are for saving merged models or GGUF, not typically run in a sequential script without specific intent.
# Merge to 16bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",) # This might be redundant if save_lora is used
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# Save to 8bit Q8_0
# if False: model.save_pretrained_gguf("model", tokenizer,)
# if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options
# if False:
#     model.push_to_hub_gguf(
#         "hf/model",
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
#         token = "",
#     )

print("Script conversion complete. GRPO training and LoRA saving are included.")
print("SFT training and example generation cells are commented out but can be re-enabled.")
print("Final model saving options (merged, GGUF) are also commented out.")
