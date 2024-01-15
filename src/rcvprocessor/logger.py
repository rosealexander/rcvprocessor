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

import logging
import os
import sys
from contextlib import contextmanager

LEVELS = dict(
    CRITICAL=logging.CRITICAL,
    ERROR=logging.ERROR,
    WARNING=logging.WARNING,
    INFO=logging.INFO,
    DEBUG=logging.DEBUG,
    NOTSET=logging.NOTSET,
)


@contextmanager
def logging_context(logger, level):
    log = logging.getLogger(logger)
    lvl = log.getEffectiveLevel()
    log.setLevel(LEVELS.get(level.upper(), "WARN"))
    yield
    log.setLevel(lvl)


class Logger:
    def __init__(self, name="KVS_sagemaker_integration"):
        self.logger = _create_logger(name)

    def get(self):
        return self.logger


def _default_level():
    level = os.getenv("LOG_LEVEL", "WARN")
    return LEVELS.get(level.upper(), "WARN")


def _create_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger.setLevel(_default_level())
    return logger
