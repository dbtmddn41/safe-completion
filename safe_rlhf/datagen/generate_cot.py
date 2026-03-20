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
"""Generate answer + Chain-of-Thought (CoT) from prompt-centric JSONL inputs.

Usage:
    python -m safe_rlhf.datagen generate \
        --model_name_or_path <model> \
        --input_file <prompts.jsonl> \
        --output_file <generated_output.jsonl>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from safe_rlhf.models.pretrained import suppress_tie_weights_mapping_warning


__all__ = ['main']

DEFAULT_COT_SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Given a prompt and optional extra context, solve the task and output both reasoning and final answer.\n\n"
    "You must return strictly in this format:\n"
    "<reasoning>\n"
    "...your step-by-step reasoning...\n"
    "</reasoning>\n"
    "<answer>\n"
    "...your final answer...\n"
    "</answer>"
)

DEFAULT_COT_USER_TEMPLATE = (
    "Prompt:\n{prompt}\n\n"
    "Additional Information (JSON):\n{extra_json}\n\n"
    "Solve the task and provide both reasoning and answer with the required tags."
)

FALLBACK_PROMPT_INPUT = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:'


class DefaultFormatDict(dict[str, Any]):
    """Default dict for safe template substitution."""

    def __missing__(self, key: str) -> str:
        return ''


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='python -m safe_rlhf.datagen generate',
        description='Stage 1: Generate CoT data using an LLM.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model arguments
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='HuggingFace model name or local path for CoT generation.',
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
            'Path to input JSONL file. Each line must be a JSON object with at least '
            'a prompt field (default: "input"). Other fields are kept and can be templated.'
        ),
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output JSONL file for generated CoT data.',
    )

    # Prompt arguments
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=DEFAULT_COT_SYSTEM_PROMPT,
        help='System prompt to guide CoT generation.',
    )
    parser.add_argument(
        '--user_template',
        type=str,
        default=DEFAULT_COT_USER_TEMPLATE,
        help=(
            'User message template. Supports all input keys plus {prompt} and {extra_json}. '
            'Unknown placeholders are replaced with empty string.'
        ),
    )
    parser.add_argument(
        '--prompt_field',
        type=str,
        default='input',
        help='Field name in each JSON object to use as prompt.',
    )

    # Generation arguments
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling.')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling.')
    parser.add_argument('--do_sample', type=bool, default=True, help='Whether to use sampling.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of CoT sequences per input.')
    parser.add_argument('--batch_size', type=int, default=4, help='Inference batch size.')
    parser.add_argument(
        '--max_batch_tokens',
        type=int,
        default=12288,
        help='Approximate token budget per batch for dynamic batching.',
    )
    parser.add_argument('--max_prompt_tokens', type=int, default=3072, help='Max prompt tokens before generation.')
    parser.add_argument(
        '--backend',
        type=str,
        default='transformers',
        choices=['transformers', 'vllm'],
        help='Inference backend to use.',
    )
    parser.add_argument(
        '--vllm_tensor_parallel_size',
        type=int,
        default=1,
        help='Tensor parallel size for vLLM backend.',
    )
    parser.add_argument(
        '--vllm_gpu_memory_utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization for vLLM backend.',
    )
    parser.add_argument(
        '--vllm_max_num_batched_tokens',
        type=int,
        default=None,
        help='Override vLLM max_num_batched_tokens (optional).',
    )
    parser.add_argument(
        '--vllm_max_model_len',
        type=int,
        default=None,
        help=(
            'Override max model context length for vLLM. '
            'If not set, defaults to max_prompt_tokens + max_new_tokens '
            'to avoid allocating KV cache for the full pretrained max context.'
        ),
    )

    # Device arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, cuda:0, ...).')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16', 'fp32'], help='Data type for inference.')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8-bit quantization.')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit quantization.')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file (skip already generated IDs).')

    return parser.parse_args()


def load_input_data(input_file: str, prompt_field: str) -> list[dict[str, Any]]:
    """Load input data from a JSONL file."""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if 'id' not in item:
                item['id'] = i
            if prompt_field not in item:
                raise ValueError(
                    f"Line {i} missing required prompt field '{prompt_field}': {line[:100]}"
                )
            data.append(item)
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
    user_content: str,
) -> str:
    """Build a chat prompt using the tokenizer's chat template if available."""
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


def resolve_vllm_dtype(dtype_str: str) -> str:
    """Resolve dtype string to vLLM dtype string."""
    mapping = {
        'auto': 'auto',
        'fp16': 'float16',
        'bf16': 'bfloat16',
        'fp32': 'float32',
    }
    return mapping.get(dtype_str, 'auto')


def render_user_content(user_template: str, item: dict[str, Any], prompt_field: str) -> str:
    """Render user content from arbitrary input schema."""
    prompt = str(item[prompt_field])
    extra = {key: value for key, value in item.items() if key not in {prompt_field, 'id'}}
    format_values: dict[str, Any] = dict(item)
    format_values['prompt'] = prompt
    format_values['input'] = prompt
    format_values['extra_json'] = json.dumps(extra, ensure_ascii=False)
    return user_template.format_map(DefaultFormatDict(format_values))


def estimate_prompt_lengths(
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_prompt_tokens: int,
) -> list[int]:
    """Estimate token lengths for dynamic batching."""
    lengths = []
    for prompt in prompts:
        tokenized = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.append(len(tokenized['input_ids']))
    return lengths


def build_dynamic_batches(
    lengths: list[int],
    batch_size: int,
    max_batch_tokens: int,
) -> list[list[int]]:
    """Build dynamic batches with length bucketing + token budget."""
    if not lengths:
        return []

    sorted_indices = sorted(range(len(lengths)), key=lengths.__getitem__)
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_max_len = 0

    for index in sorted_indices:
        candidate_len = lengths[index]
        next_max_len = max(current_max_len, candidate_len)
        next_batch_size = len(current_batch) + 1
        estimated_tokens = next_max_len * next_batch_size

        if current_batch and (
            next_batch_size > batch_size or estimated_tokens > max_batch_tokens
        ):
            batches.append(current_batch)
            current_batch = [index]
            current_max_len = candidate_len
            continue

        current_batch.append(index)
        current_max_len = next_max_len

    if current_batch:
        batches.append(current_batch)

    return batches


def parse_reasoning_and_answer(text: str) -> tuple[str, str]:
    """Parse reasoning and answer from tagged generation output."""
    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', text, flags=re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.IGNORECASE | re.DOTALL)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
    answer = answer_match.group(1).strip() if answer_match else ''

    if not reasoning and not answer:
        answer_split = re.split(r'\n\s*answer\s*:\s*', text, flags=re.IGNORECASE, maxsplit=1)
        if len(answer_split) == 2:
            reasoning = answer_split[0].strip()
            answer = answer_split[1].strip()
        else:
            answer = text.strip()

    return reasoning, answer


def build_result_records(
    input_data: list[dict[str, Any]],
    batch_indices: list[int],
    grouped_outputs: list[list[str]],
    prompt_field: str,
) -> dict[int, dict[str, Any]]:
    """Convert grouped generation outputs into result records keyed by input index."""
    generated_by_index: dict[int, dict[str, Any]] = {}
    for local_index, data_index in enumerate(batch_indices):
        item = input_data[data_index]
        outputs_for_item = grouped_outputs[local_index]

        cot_values = []
        answer_values = []
        for generated_text in outputs_for_item:
            reasoning, answer = parse_reasoning_and_answer(generated_text)
            cot_values.append(reasoning)
            answer_values.append(answer)

        result: dict[str, Any] = dict(item)
        result['prompt'] = str(item[prompt_field])
        if len(outputs_for_item) > 1:
            result['cot'] = cot_values
            result['answer'] = answer_values
            result['raw_generation'] = outputs_for_item
        else:
            result['cot'] = cot_values[0] if cot_values else ''
            result['answer'] = answer_values[0] if answer_values else ''
            result['raw_generation'] = outputs_for_item[0] if outputs_for_item else ''

        generated_by_index[data_index] = result
    return generated_by_index


def main() -> None:
    args = parse_arguments()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    print("===== Prompt → CoT + Answer Generation =====")
    print(f"Model       : {args.model_name_or_path}")
    print(f"Input file  : {args.input_file}")
    print(f"Output file : {args.output_file}")
    print(f"Prompt field: {args.prompt_field}")
    print(f"Backend     : {args.backend}")

    # ── Load input data ──────────────────────────────────────────────────────
    input_data = load_input_data(args.input_file, args.prompt_field)
    print(f"Loaded {len(input_data)} input samples.")

    # ── Resume support ───────────────────────────────────────────────────────
    if args.resume:
        existing_ids = load_existing_ids(args.output_file)
        input_data = [item for item in input_data if item['id'] not in existing_ids]
        print(f"Resuming: {len(existing_ids)} already done, {len(input_data)} remaining.")

    if not input_data:
        print("Nothing to generate. Exiting.")
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

    model = None
    llm = None
    device = None
    gen_config = None

    if args.backend == 'transformers':
        print("Loading model (transformers backend)...")
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
            device = torch.device(
                args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
            )
            model = model.to(device)
        else:
            device = next(model.parameters()).device

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        try:
            from vllm import LLM
        except ImportError as ex:
            raise ImportError(
                "vLLM backend requested but 'vllm' is not installed. "
                "Install with: pip install vllm"
            ) from ex

        print("Loading model (vLLM backend)...")
        effective_vllm_max_model_len = args.vllm_max_model_len
        if effective_vllm_max_model_len is None:
            effective_vllm_max_model_len = args.max_prompt_tokens + args.max_new_tokens

        llm_kwargs: dict[str, Any] = {
            'model': args.model_name_or_path,
            'trust_remote_code': args.trust_remote_code,
            'dtype': resolve_vllm_dtype(args.dtype),
            'tensor_parallel_size': args.vllm_tensor_parallel_size,
            'gpu_memory_utilization': args.vllm_gpu_memory_utilization,
            'max_model_len': effective_vllm_max_model_len,
        }
        if args.vllm_max_num_batched_tokens is not None:
            llm_kwargs['max_num_batched_tokens'] = args.vllm_max_num_batched_tokens

        print(f"vLLM max_model_len: {effective_vllm_max_model_len}")
        llm = LLM(**llm_kwargs)

    # ── Output directory ─────────────────────────────────────────────────────
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ── Build prompts once and create dynamic batches ───────────────────────
    prompts = [
        build_prompt(
            tokenizer,
            args.system_prompt,
            render_user_content(args.user_template, item, args.prompt_field),
        )
        for item in input_data
    ]
    prompt_lengths = estimate_prompt_lengths(tokenizer, prompts, args.max_prompt_tokens)
    batches = build_dynamic_batches(
        prompt_lengths,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    print(f"Dynamic batches: {len(batches)} (max_batch_size={args.batch_size}, max_batch_tokens={args.max_batch_tokens})")

    generated_by_index: dict[int, dict[str, Any]] = {}

    # ── Generate CoT + Answer ───────────────────────────────────────────────
    open_mode = 'a' if args.resume else 'w'
    for batch_indices in tqdm(batches, desc=f'Generating CoT+Answer ({args.backend})'):
        batch_prompts = [prompts[index] for index in batch_indices]

        if args.backend == 'transformers':
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
            ).to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    generation_config=gen_config,
                )

            input_lengths = [int(x) for x in inputs['attention_mask'].sum(dim=1).tolist()]
            expanded_input_lengths: list[int] = []
            for length in input_lengths:
                expanded_input_lengths.extend([length] * args.num_return_sequences)

            generated_texts: list[str] = []
            for row_index, input_length in enumerate(expanded_input_lengths):
                generated_tokens = outputs[row_index, input_length:]
                generated_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))

            grouped_outputs = []
            for index in range(len(batch_indices)):
                seq_start = index * args.num_return_sequences
                seq_end = seq_start + args.num_return_sequences
                grouped_outputs.append(generated_texts[seq_start:seq_end])
        else:
            from vllm import SamplingParams

            temperature = args.temperature if args.do_sample else 0.0
            top_p = args.top_p if args.do_sample else 1.0
            top_k = args.top_k if args.do_sample else -1
            sampling_params = SamplingParams(
                n=args.num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=args.max_new_tokens,
                seed=args.seed,
            )

            vllm_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            grouped_outputs = []
            for output in vllm_outputs:
                grouped_outputs.append([candidate.text for candidate in output.outputs])

        generated_by_index.update(
            build_result_records(
                input_data=input_data,
                batch_indices=batch_indices,
                grouped_outputs=grouped_outputs,
                prompt_field=args.prompt_field,
            )
        )

    with open(args.output_file, open_mode, encoding='utf-8') as fout:
        for data_index in range(len(input_data)):
            fout.write(json.dumps(generated_by_index[data_index], ensure_ascii=False) + '\n')

    print(f"Done! Generated answer+CoT for {len(input_data)} samples → {args.output_file}")


if __name__ == '__main__':
    main()
