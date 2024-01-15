# Copyright 2024 Alexander Rose
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bisect import insort
from dataclasses import dataclass
from operator import attrgetter

import numpy as np


@dataclass(kw_only=True)
class Frame:
    id: str
    img: np.ndarray
    labels: list
    time: [int, float]

    def __str__(self):
        return str(dict(
            id=self.id,
            img=type(self.img),
            labels=self.labels,
            time=self.time
        ))


class FrameContainer:
    def __init__(self):
        self._inner_list = list()

    def __delitem__(self, index):
        self._inner_list.__delitem__(index)

    def __getitem__(self, index):
        return self._inner_list.__getitem__(index)

    def __len__(self):
        return len(self._inner_list)

    def __iter__(self):
        yield from self._inner_list

    def __str__(self):
        return '\n'.join([str(c) for c in self._inner_list])

    def __bool__(self):
        return True if self._inner_list else False

    def insert(self, frame: Frame) -> None:
        insort(self._inner_list, frame, key=attrgetter('id'))
