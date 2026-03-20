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
"""PKU-SafeRLHF-QA datasets."""

from __future__ import annotations

from typing import ClassVar

from datasets import load_dataset
from safe_rlhf.datasets.base import RawDataset, RawSample


__all__ = [
    'SafeRLHFQADataset',
    'SafeRLHFQATrainDataset',
    'SafeRLHFQATestDataset',
    'SafeRLHFQASafeTrainDataset',
]


class SafeRLHFQADataset(RawDataset):
    SPLIT: ClassVar[str]

    def __init__(self, path: str | None = None) -> None:
        self.data = load_dataset(path or 'PKU-Alignment/PKU-SafeRLHF-QA', split=self.SPLIT)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        return RawSample(
            input=data['prompt'],
            answer=data['response'],
            is_safe=bool(data.get('is_safe', True)),
        )

    def __len__(self) -> int:
        return len(self.data)


class SafeRLHFQATrainDataset(SafeRLHFQADataset):
    NAME: str = 'PKU-SafeRLHF-QA/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-QA/train',)
    SPLIT: str = 'train'


class SafeRLHFQATestDataset(SafeRLHFQADataset):
    NAME: str = 'PKU-SafeRLHF-QA/test'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-QA/test',)
    SPLIT: str = 'test'


class SafeRLHFQASafeTrainDataset(SafeRLHFQADataset):
    NAME: str = 'PKU-SafeRLHF-QA-Safe/train'
    ALIASES: tuple[str, ...] = ('PKU-Alignment/PKU-SafeRLHF-QA-Safe/train',)
    SPLIT: str = 'train'

    def __init__(self, path: str | None = None) -> None:
        super().__init__(path=path)
        self.data = self.data.filter(lambda sample: bool(sample['is_safe']))