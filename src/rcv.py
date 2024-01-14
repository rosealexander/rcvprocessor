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

import base64
import itertools
import json
import time
import asyncio
from collections import deque
from dataclasses import dataclass

import cv2
from kvsparser import Parser

from .logger import Logger
from .frame import Frame, FrameContainer
from .boto import KinesisVideoMediaClient, LambdaClient, invoke_async

logger = Logger(__name__).get()


class Processor:
    def __init__(self, function_name, stream_name, **kwargs):
        self._function_name = function_name
        self._stream_name = stream_name
        self._concurrency = kwargs.get('concurrency', 60)
        self._inference_tasks = set()
        self._background_tasks = set()
        self._fragment_count = itertools.count(1)
        self._fragments = deque()
        self._stop_running = None

    @property
    def function_name(self):
        return self._function_name

    @function_name.setter
    def function_name(self, function_name):
        self._function_name = function_name

    @property
    def concurrency(self):
        return self._concurrency

    @concurrency.setter
    def concurrency(self, concurrency):
        self._concurrency = concurrency

    def __aiter__(self):
        task = asyncio.create_task(self._run_processor_async())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        self._stop_running = False
        return self

    async def __anext__(self):
        while True:
            if self._stop_running:
                break
            if self._fragments and self._fragments[0].status == 'COMPLETE':
                return self._fragments.popleft().fragment
            else:
                await asyncio.sleep(0)

        raise StopAsyncIteration

    def stop(self):
        self._stop_running = True

    async def _run_processor_async(self):
        async with KinesisVideoMediaClient(self._stream_name) as video_client:
            media_response = await video_client.get_media(
                StreamName=self._stream_name,
                StartSelector={'StartSelectorType': 'NOW'}
            )

            async for fragment in Parser(media_response):
                if self._stop_running:
                    break
                task = asyncio.create_task(self._process_fragment(fragment))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            self.stop()

    async def _process_fragment(self, fragment):
        start_time = time.time()

        fragment_idx = next(self._fragment_count)

        try:
            logger.info(f'fragment {fragment_idx}')

            tags = fragment.tags

            images = fragment.images

            if not max(0, self.concurrency - len(self._inference_tasks)):
                images.clear()

            if images:
                while len(images) > self.concurrency - len(self._inference_tasks):
                    images = images[1::3]

            frames = FrameContainer()

            for idx, image in enumerate(images):
                frames.insert(Frame(
                    id=(str(fragment_idx) + str(idx).zfill(4)).zfill(12),
                    img=image,
                    labels=list(),
                    time=0
                ))

            rcv_fragment = Fragment(id=fragment_idx, frames=frames, tags=tags)

            rcv_fragment_wrapper = FragmentWrapper(fragment=rcv_fragment, status='PENDING')

            self._fragments.append(rcv_fragment_wrapper)

            async with LambdaClient() as lambda_client:

                tasks = [asyncio.create_task(self._inference_coro(lambda_client, frame)) for frame in rcv_fragment.frames]

                for task in tasks:
                    self._inference_tasks.add(task)
                    task.add_done_callback(self._inference_tasks.discard)

                labeled_frames = await asyncio.gather(*tasks)

            labeled_frame_container = FrameContainer()

            for frame in labeled_frames:
                labeled_frame_container.insert(frame)

            rcv_fragment_wrapper.fragment.frames = labeled_frame_container

            rcv_fragment_wrapper.status = 'COMPLETE'
        except Exception as err:
            logger.error(f"{type(err).__name__}: {err}", exc_info=True)
            self.stop()
        finally:
            logger.debug(f'fragment {fragment_idx} complete in {time.time() - start_time}s')

    async def _inference_coro(self, lambda_client, frame):
        function_name = self.function_name

        encoded, buf = cv2.imencode('.jpg', cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB))
        body = buf.tobytes()
        payload = json.dumps({
            'body': base64.b64encode(body).decode("utf8"),
        })
        labels = ()
        inference_start_time = time.perf_counter_ns()
        try:
            labels = await invoke_async(lambda_client, function_name, payload)
        except Exception as err:
            # logger.error(f"{type(err).__name__}: {err}", exc_info=True)
            pass
        inference_dt = time.perf_counter_ns() - inference_start_time

        return Frame(id=frame.id, img=frame.img, labels=labels, time=inference_dt)


#
@dataclass(kw_only=True)
class Fragment:
    id: str
    frames: FrameContainer
    tags: dict
    bytes = bytearray

    def __str__(self):
        return str(dict(
            id=self.id,
            frames=self.frames,
            tags=self.tags,
        ))


#
@dataclass(kw_only=True)
class FragmentWrapper:
    fragment: Fragment
    status: str

    def __str__(self):
        return str(dict(
            fragment=self.fragment.__class__,
            status=self.status
        ))
