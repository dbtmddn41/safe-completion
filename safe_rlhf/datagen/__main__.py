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
"""Entry point for data generation pipeline."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m safe_rlhf.datagen',
        description='Data generation: prompt -> answer + CoT.',
    )
    subparsers = parser.add_subparsers(dest='stage', help='Pipeline stage to run.')

    subparsers.add_parser('generate', help='Generate answer + CoT from prompts.')
    subparsers.add_parser('evaluate', help='Legacy evaluator (optional).')
    subparsers.add_parser('pipeline', help='Single-stage generation pipeline (recommended).')

    args, remaining = parser.parse_known_args()

    if args.stage == 'generate':
        from safe_rlhf.datagen.generate_cot import main as generate_main
        sys.argv = [sys.argv[0]] + remaining
        generate_main()
    elif args.stage == 'evaluate':
        from safe_rlhf.datagen.evaluate_cot import main as evaluate_main
        sys.argv = [sys.argv[0]] + remaining
        evaluate_main()
    elif args.stage == 'pipeline':
        from safe_rlhf.datagen.pipeline import main as pipeline_main
        sys.argv = [sys.argv[0]] + remaining
        pipeline_main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
