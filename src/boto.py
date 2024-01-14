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

import json
from contextlib import AsyncExitStack

from aiobotocore.config import AioConfig
from aiobotocore.session import AioSession

from .logger import Logger, logging_context

logger = Logger(__name__).get()


class KinesisVideoClient:
    def __init__(self, stream_name, **kwargs):
        self._stream_name = stream_name
        self._exit_stack = AsyncExitStack()
        self._kvs_client = None

    @property
    def stream_name(self):
        return self._stream_name

    @stream_name.setter
    def stream_name(self, stream_name):
        self._stream_name = stream_name

    async def __aenter__(self):
        kvs_client = AioSession().create_client('kinesisvideo')

        with logging_context('aiobotocore.credentials', 'WARN'):
            self._kvs_client = await self._exit_stack.enter_async_context(kvs_client)

        return self._kvs_client

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self._exit_stack.__aexit__(exc_t, exc_v, exc_tb)
        self._kvs_client = None


class KinesisVideoMediaClient:
    def __init__(self, stream_name, **kwargs):
        self._stream_name = stream_name
        self._exit_stack = AsyncExitStack()
        self._kvs_media_client = None

    @property
    def stream_name(self):
        return self._stream_name

    @stream_name.setter
    def stream_name(self, stream_name):
        self._stream_name = stream_name

    async def __aenter__(self):

        async with KinesisVideoClient(self._stream_name) as kvs_client:
            kvs_endpoint = await kvs_client.get_data_endpoint(StreamName=self._stream_name, APIName='GET_MEDIA')
            endpoint_url = kvs_endpoint.get('DataEndpoint')

        kvs_media_client = AioSession().create_client('kinesis-video-media', endpoint_url=endpoint_url)

        with logging_context('aiobotocore.credentials', 'WARN'):
            self._kvs_media_client = await self._exit_stack.enter_async_context(kvs_media_client)

        return self._kvs_media_client

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self._exit_stack.__aexit__(exc_t, exc_v, exc_tb)
        self._kvs_media_client = None


class LambdaClient:
    def __init__(self, **kwargs):
        self._max_pool_connections = int(kwargs.get('max_pool_connections') or '120')
        self._connect_timeout = int(kwargs.get('connect_timeout') or '5')
        self._read_timeout = int(kwargs.get('read_timeout') or '60')
        self._retries = int(kwargs.get('retries') or '20')
        self._exit_stack = AsyncExitStack()
        self._lambda_client = None

    @property
    def max_pool_connections(self):
        return self._max_pool_connections

    @max_pool_connections.setter
    def max_pool_connections(self, max_pool_connections):
        self._max_pool_connections = max_pool_connections

    @property
    def connect_timeout(self):
        return self._connect_timeout

    @connect_timeout.setter
    def connect_timeout(self, connect_timeout):
        self._connect_timeout = connect_timeout

    @property
    def read_timeout(self):
        return self._read_timeout

    @read_timeout.setter
    def read_timeout(self, read_timeout):
        self._read_timeout = read_timeout

    @property
    def retries(self):
        return self._retries

    @retries.setter
    def retries(self, retries):
        self._retries = retries

    async def __aenter__(self):
        lambda_client = self._create_client()

        with logging_context('aiobotocore.credentials', 'WARN'):
            self._lambda_client = await self._exit_stack.enter_async_context(lambda_client)

        return self._lambda_client

    async def __aexit__(self, exc_t, exc_v, exc_tb):
        await self._exit_stack.__aexit__(exc_t, exc_v, exc_tb)

    def _create_client(self):
        config = AioConfig(
            max_pool_connections=self.max_pool_connections,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            retries={'max_attempts': self.retries}
        )

        return AioSession().create_client('lambda', config=config)


async def invoke_async(lambda_client, function_name, payload):
    res = await lambda_client.invoke(
        FunctionName=function_name,
        Payload=payload
    )
    body = await res['Payload'].read()
    return json.loads(body).get('body')
