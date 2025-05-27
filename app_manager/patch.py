
from starlette.responses import StreamingResponse
from starlette.types import Receive
import asyncio

class SilentStreamingResponse(StreamingResponse):
    async def listen_for_disconnect(self, receive: Receive) -> None:
        try:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    break
        except asyncio.CancelledError:
            pass  