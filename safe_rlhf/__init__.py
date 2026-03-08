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
"""Safe-RLHF: Safe Reinforcement Learning with Human Feedback."""

from __future__ import annotations

import importlib
from typing import Any

from safe_rlhf.version import __version__


_SUBMODULES = (
    'algorithms',
    'configs',
    'datasets',
    'models',
    'trainers',
    'utils',
    'values',
)

__all__ = ['__version__', *_SUBMODULES]


def __getattr__(name: str) -> Any:
    """Lazily import top-level subpackages and exported symbols.

    This avoids importing heavy training dependencies such as DeepSpeed when a
    lightweight utility module (for example `safe_rlhf.datagen`) is executed.
    """
    if name in _SUBMODULES:
        module = importlib.import_module(f'safe_rlhf.{name}')
        globals()[name] = module
        return module

    for module_name in _SUBMODULES:
        module = importlib.import_module(f'safe_rlhf.{module_name}')
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
