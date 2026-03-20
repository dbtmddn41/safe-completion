# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Stage 2: Evaluate Chain-of-Thought (CoT) data against pre-made labels.

Usage:
    python -m safe_rlhf.datagen evaluate \
        --model_name_or_path <model> \
        --input_file <cot_output.jsonl> \
        --label_file <labels.jsonl> \
        --output_file <eval_output.jsonl> \
        [--max_new_tokens 512] \
        [--temperature 0.1] \
        [--batch_size 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from safe_rlhf.models.pretrained import suppress_tie_weights_mapping_warning


__all__ = ['main']

DEFAULT_EVAL_SYSTEM_PROMPT = (
    "You are an expert evaluator. You will be given a problem, a chain-of-thought "
    "reasoning, and a reference answer (ground truth label). "
    "Your task is to evaluate the quality of the chain-of-thought reasoning.\n\n"
    "Evaluate on the following criteria:\n"
    "1. Correctness: Does the reasoning lead to the correct answer?\n"
    "2. Completeness: Are all necessary reasoning steps included?\n"
    "3. Coherence: Is the reasoning logically coherent and well-structured?\n\n"
    "Provide your evaluation in the following format:\n"
    "Correctness: <correct/incorrect>\n"
    "Completeness: <complete/incomplete>\n"
    "Coherence: <coherent/incoherent>\n"
    "Score: <1-5>\n"
    "Feedback: <brief explanation>"
)

DEFAULT_EVAL_USER_TEMPLATE = (
    "Please evaluate the following chain-of-thought reasoning.\n\n"
    "Problem: {input}\n\n"
    "Chain-of-Thought Reasoning:\n{cot}\n\n"
    "Reference Answer (Ground Truth): {label}\n\n"
    "Provide your evaluation."
)

FALLBACK_PROMPT_INPUT = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:'


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='python -m safe_rlhf.datagen evaluate',
        description='Stage 2: Evaluate CoT data against reference labels using an LLM.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model arguments
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='HuggingFace model name or local path for CoT evaluation.',
    )
    parser.add_argument(
        '--trust_remote_code',
        type=bool,
        default=True,
        help='Whether to trust remote code when loading the model.',
    )

    # Data arguments
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help=(
            'Path to CoT JSONL file (output of stage 1). '
            'Each line must have "id", "input", and "cot" fields.'
        ),
    )
    parser.add_argument(
        '--label_file',
        type=str,
        default=None,
        help=(
            'Path to label JSONL file. Each line must have "id" and "label" fields. '
            'If not provided, labels are expected in the input_file itself.'
        ),
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output JSONL file for evaluation results.',
    )

    # Prompt arguments
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=DEFAULT_EVAL_SYSTEM_PROMPT,
        help='System prompt to guide evaluation.',
    )
    parser.add_argument(
        '--user_template',
        type=str,
        default=DEFAULT_EVAL_USER_TEMPLATE,
        help='User message template. Must contain {input}, {cot}, and {label} placeholders.',
    )

    # Generation arguments
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature (low for deterministic eval).')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling.')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling.')
    parser.add_argument('--do_sample', type=bool, default=True, help='Whether to use sampling.')
    parser.add_argument('--batch_size', type=int, default=4, help='Inference batch size.')

    # Device arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, cuda:0, ...).')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16', 'fp32'], help='Data type for inference.')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8-bit quantization.')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit quantization.')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file.')

    return parser.parse_args()


def load_jsonl(filepath: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_existing_ids(output_file: str) -> set:
    """Load already-processed IDs from an existing output file for resumption."""
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                existing_ids.add(item.get('id'))
    return existing_ids


def build_prompt(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_template: str,
    input_text: str,
    cot_text: str,
    label_text: str,
) -> str:
    """Build a chat prompt for evaluation."""
    user_content = user_template.format(input=input_text, cot=cot_text, label=label_text)

    # Try to use the tokenizer's chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content},
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt
        except Exception:
            pass

    # Fallback: simple concatenation
    merged_content = user_content.strip()
    if system_prompt.strip():
        merged_content = f"[System]\n{system_prompt.strip()}\n\n{merged_content}"
    return FALLBACK_PROMPT_INPUT.format(input=merged_content)


def resolve_dtype(dtype_str: str) -> torch.dtype | str:
    """Resolve dtype string to torch dtype."""
    mapping = {
        'auto': 'auto',
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    return mapping.get(dtype_str, 'auto')


def main() -> None:
    args = parse_arguments()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"===== CoT Evaluation (Stage 2) =====")
    print(f"Model       : {args.model_name_or_path}")
    print(f"Input file  : {args.input_file}")
    print(f"Label file  : {args.label_file or '(labels from input file)'}")
    print(f"Output file : {args.output_file}")

    # ── Load CoT data ────────────────────────────────────────────────────────
    cot_data = load_jsonl(args.input_file)
    print(f"Loaded {len(cot_data)} CoT samples.")

    # ── Load labels ──────────────────────────────────────────────────────────
    if args.label_file:
        label_data = load_jsonl(args.label_file)
        label_map = {item['id']: item['label'] for item in label_data}
        # Merge labels into cot_data
        for item in cot_data:
            item_id = item.get('id')
            if item_id in label_map:
                item['label'] = label_map[item_id]
            elif 'label' not in item:
                raise ValueError(
                    f"No label found for id={item_id}. "
                    "Provide labels in label_file or in the input_file."
                )
    else:
        # Expect labels in input_file
        for item in cot_data:
            if 'label' not in item:
                raise ValueError(
                    f"No label found for id={item.get('id')}. "
                    "Provide --label_file or include 'label' field in input_file."
                )

    # ── Resume support ───────────────────────────────────────────────────────
    if args.resume:
        existing_ids = load_existing_ids(args.output_file)
        cot_data = [item for item in cot_data if item['id'] not in existing_ids]
        print(f"Resuming: {len(existing_ids)} already done, {len(cot_data)} remaining.")

    if not cot_data:
        print("Nothing to evaluate. Exiting.")
        return

    # ── Load model and tokenizer ─────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        padding_side='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model_kwargs: dict[str, Any] = {
        'trust_remote_code': args.trust_remote_code,
        'torch_dtype': resolve_dtype(args.dtype),
    }
    if args.device == 'auto':
        model_kwargs['device_map'] = 'auto'
    if args.load_in_8bit:
        model_kwargs['load_in_8bit'] = True
    if args.load_in_4bit:
        model_kwargs['load_in_4bit'] = True

    with suppress_tie_weights_mapping_warning():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )
    model.eval()

    if args.device != 'auto' and not (args.load_in_8bit or args.load_in_4bit):
        device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    else:
        device = next(model.parameters()).device

    # ── Generation config ────────────────────────────────────────────────────
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ── Output directory ─────────────────────────────────────────────────────
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Evaluate CoT ─────────────────────────────────────────────────────────
    open_mode = 'a' if args.resume else 'w'
    with open(args.output_file, open_mode, encoding='utf-8') as fout:
        for batch_start in tqdm(range(0, len(cot_data), args.batch_size), desc='Evaluating CoT'):
            batch = cot_data[batch_start : batch_start + args.batch_size]

            # Build prompts
            prompts = []
            for item in batch:
                cot_text = item['cot']
                if isinstance(cot_text, list):
                    cot_text = cot_text[0]  # Use first CoT if multiple were generated
                prompts.append(
                    build_prompt(
                        tokenizer,
                        args.system_prompt,
                        args.user_template,
                        input_text=item['input'],
                        cot_text=cot_text,
                        label_text=str(item['label']),
                    )
                )

            # Tokenize
            inputs = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                )

            # Decode
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Write results
            for i, item in enumerate(batch):
                cot_text = item['cot']
                if isinstance(cot_text, list):
                    cot_text = cot_text[0]

                result = {
                    'id': item['id'],
                    'input': item['input'],
                    'cot': cot_text,
                    'label': item['label'],
                    'evaluation': generated_texts[i],
                }
                # Carry forward any extra fields
                for key in item:
                    if key not in result:
                        result[key] = item[key]

                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()

    print(f"Done! Evaluated {len(cot_data)} samples → {args.output_file}")


if __name__ == '__main__':
    main()
