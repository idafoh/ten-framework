import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..config import BytedanceASRLLMConfig
from ..extension import BytedanceASRLLMExtension


def _minimal_config() -> BytedanceASRLLMConfig:
    return BytedanceASRLLMConfig.model_validate(
        {
            "params": {
                "audio": {"rate": 16000},
                "request": {"model_name": "bigmodel"},
            }
        }
    )


class _RaceClient:
    def __init__(
        self,
        audio_send_started: asyncio.Event,
        allow_audio_send: asyncio.Event,
    ) -> None:
        self.connected = True
        self._audio_send_started = audio_send_started
        self._allow_audio_send = allow_audio_send
        self.sent_audio: list[bytes] = []

    async def send_audio(self, audio_data: bytes) -> None:
        self._audio_send_started.set()
        await self._allow_audio_send.wait()
        if not self.connected:
            raise RuntimeError("Not connected to ASR service")
        self.sent_audio.append(audio_data)


@pytest.mark.asyncio
async def test_update_configs_waits_for_in_flight_audio_before_reconnecting():
    audio_send_started = asyncio.Event()
    allow_audio_send = asyncio.Event()
    client = _RaceClient(
        audio_send_started,
        allow_audio_send,
    )

    extension = BytedanceASRLLMExtension("test_extension")
    extension.ten_env = MagicMock()
    extension.config = _minimal_config()
    extension.connected = True
    extension.client = client
    extension.send_asr_error = AsyncMock()

    async def stop_connection() -> None:
        client.connected = False
        extension.connected = False

    async def start_connection() -> None:
        client.connected = True
        extension.connected = True

    extension.stop_connection = AsyncMock(side_effect=stop_connection)
    extension.start_connection = AsyncMock(side_effect=start_connection)

    frame = MagicMock()
    frame.lock_buf.return_value = b"\x00\x01"

    audio_task = asyncio.create_task(extension.send_audio(frame, "session-1"))
    await audio_send_started.wait()
    update_task = asyncio.create_task(
        extension._run_update_configs(
            {
                "params": {
                    "request": {
                        "corpus": {"context": "updated dialog context"},
                    }
                }
            }
        )
    )
    await asyncio.sleep(0)
    extension.stop_connection.assert_not_awaited()
    allow_audio_send.set()

    update_result = await update_task
    audio_result = await audio_task

    assert update_result == (True, "")
    assert audio_result is True
    assert extension.is_connected() is True
    assert client.sent_audio == [b"\x00\x01"]
    extension.send_asr_error.assert_not_awaited()
