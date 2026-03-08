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
"""Run single-stage data generation (prompt -> answer + CoT)."""

from __future__ import annotations

import argparse
import os
import sys


__all__ = ['main']


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for prompt -> answer+CoT pipeline."""
    parser = argparse.ArgumentParser(
        prog='python -m safe_rlhf.datagen pipeline',
        description='Run single-stage CoT+answer generation pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        '--model',
        '--generate_model',
        dest='model',
        type=str,
        required=True,
        help='HuggingFace model name or path for generation.',
    )
    parser.add_argument(
        '--trust_remote_code',
        type=bool,
        default=True,
        help='Whether to trust remote code when loading models.',
    )

    # Data arguments
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to input JSONL file with prompts.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to store outputs.',
    )

    # Generation arguments
    parser.add_argument('--prompt_field', type=str, default='input', help='Field name to use as prompt.')
    parser.add_argument('--gen_max_new_tokens', type=int, default=1024, help='Max new tokens for generation.')
    parser.add_argument('--gen_temperature', type=float, default=0.7, help='Temperature for generation.')
    parser.add_argument('--gen_batch_size', type=int, default=4, help='Max batch size for generation.')
    parser.add_argument('--max_batch_tokens', type=int, default=12288, help='Approximate token budget per dynamic batch.')
    parser.add_argument('--max_prompt_tokens', type=int, default=3072, help='Max prompt tokens before generation.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of outputs per input.')
    parser.add_argument('--backend', type=str, default='transformers', choices=['transformers', 'vllm'], help='Inference backend.')
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1, help='Tensor parallel size for vLLM backend.')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for vLLM backend.')
    parser.add_argument('--vllm_max_num_batched_tokens', type=int, default=None, help='Override max batched tokens for vLLM backend.')
    parser.add_argument(
        '--vllm_max_model_len',
        type=int,
        default=None,
        help='Override max context length for vLLM backend.',
    )

    # Device arguments
    parser.add_argument('--device', type=str, default='auto', help='Device (auto, cpu, cuda, cuda:0, ...).')
    parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16', 'fp32'], help='Data type.')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load models in 8-bit.')
    parser.add_argument('--load_in_4bit', action='store_true', help='Load models in 4-bit.')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file.')

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'generated_answer_cot.jsonl')

    print("=" * 60)
    print("  Prompt -> Answer + CoT Pipeline")
    print("=" * 60)
    print(f"  Model                    : {args.model}")
    print(f"  Input file               : {args.input_file}")
    print(f"  Prompt field             : {args.prompt_field}")
    print(f"  Backend                  : {args.backend}")
    print(f"  Output dir               : {args.output_dir}")
    print(f"  Output file              : {output_file}")
    print("=" * 60)

    print("\n>>> Generating answer + CoT data...")
    gen_argv = [
        sys.argv[0],
        '--model_name_or_path', args.model,
        '--input_file', args.input_file,
        '--output_file', output_file,
        '--prompt_field', args.prompt_field,
        '--max_new_tokens', str(args.gen_max_new_tokens),
        '--temperature', str(args.gen_temperature),
        '--batch_size', str(args.gen_batch_size),
        '--max_batch_tokens', str(args.max_batch_tokens),
        '--max_prompt_tokens', str(args.max_prompt_tokens),
        '--num_return_sequences', str(args.num_return_sequences),
        '--backend', args.backend,
        '--vllm_tensor_parallel_size', str(args.vllm_tensor_parallel_size),
        '--vllm_gpu_memory_utilization', str(args.vllm_gpu_memory_utilization),
        '--device', args.device,
        '--dtype', args.dtype,
        '--seed', str(args.seed),
        '--trust_remote_code', str(args.trust_remote_code),
    ]
    if args.vllm_max_num_batched_tokens is not None:
        gen_argv.extend(['--vllm_max_num_batched_tokens', str(args.vllm_max_num_batched_tokens)])
    if args.vllm_max_model_len is not None:
        gen_argv.extend(['--vllm_max_model_len', str(args.vllm_max_model_len)])
    if args.load_in_8bit:
        gen_argv.append('--load_in_8bit')
    if args.load_in_4bit:
        gen_argv.append('--load_in_4bit')
    if args.resume:
        gen_argv.append('--resume')

    saved_argv = sys.argv
    sys.argv = gen_argv
    from safe_rlhf.datagen.generate_cot import main as generate_main

    generate_main()
    sys.argv = saved_argv

    print(">>> Generation complete.")

    print("\n" + "=" * 60)
    print("  Pipeline finished!")
    print(f"  Output data: {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
