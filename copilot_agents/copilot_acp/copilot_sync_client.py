import asyncio
import json
import os
import threading
import sys
from enum import Enum, auto
from typing import Dict, Any
from pathlib import Path

from loguru import logger
import copilot
from anthropic.types import TextBlock, ToolUseBlock, Message

from .transport import Transport, StdioTransport

# Configure loguru logging
logger.remove()
if os.environ.get("COPILOT_LOG", "false").lower() == "true":
    _log_path = Path(__file__).parent.parent.parent / "tmp" / "copilot.log"
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(_log_path, level="DEBUG")


class SessinStatus(Enum):
    INIT = auto()
    SENT_MESSAGE = auto()
    WAIT_TOOL_CALL = auto()
    SENT_TOOL_RESULT = auto()


class CopilotSession:
    def __init__(self, session_id: str, client: 'CopilotSyncClient'):
        self.session_id = session_id
        self._client = client
        self._status = SessinStatus.INIT
        self._queue = asyncio.Queue()

    def send_messages(self, messages) -> Message:
        # ACP doesn't support message array like OpenAI/Anthropic API, so we only look at the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg
                break

        if last_user_msg is None:
            raise ValueError("No user message found in messages")

        content = last_user_msg["content"]
        if isinstance(content, str):
            return self._send_prompt(content)

        elif isinstance(content, list): # can we process list of tool_result?
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result = self._send_tool_result(
                        tool_use_id=int(block["tool_use_id"]),
                        content=block["content"],
                        is_error=block.get("is_error", False))
                    return result

    def _send_prompt(self, prompt: str) -> Message:
        params = {
            "sessionId": self.session_id,
            "prompt": prompt
        }

        fut = asyncio.run_coroutine_threadsafe(
            self._client.send_message("session.send", params),
            self._client.loop
        )

        self._status = SessinStatus.SENT_MESSAGE
        fut.result()  # wait for send to complete

        result = self._wait()
        return result

    def _send_tool_result(self, tool_use_id, content: str | dict, is_error=False):
        if isinstance(content, dict):
            content_str = json.dumps(content)
        else:
            content_str = content

        result_type = "error" if is_error else "success"

        tool_result = {
            "textResultForLlm": content_str,
            "resultType": result_type
        }

        fut = asyncio.run_coroutine_threadsafe(
            self._client.send_result(tool_use_id, tool_result),  
            self._client.loop
        )

        self._status = SessinStatus.SENT_TOOL_RESULT
        fut.result()

        result = self._wait()
        return result

    async def handle_message(self, msg: Dict[str, Any]):
        if self._status == SessinStatus.SENT_MESSAGE or self._status == SessinStatus.SENT_TOOL_RESULT:
            if msg.get("method") == "tool.call":
                block = ToolUseBlock(type="tool_use",
                                       id=str(msg["id"]),
                                       name=msg["params"]["toolName"],
                                       input={**msg["params"]["arguments"]})
                message = self._build_athropic_message(block, stop_reason="tool_use")
                await self._queue.put(message)
                self._status = SessinStatus.WAIT_TOOL_CALL

            elif msg.get("method") == "session.event":
                event = msg.get("params", {}).get("event", {})
                if event.get("type") == "assistant.message":
                    data = event.get("data", {})
                    if len(data.get("toolRequests", [])) == 0 and len(data.get("content", "")) > 0:
                        # ahh final message
                        block = TextBlock(type="text", text=data.get("content"))
                        message = self._build_athropic_message(block, stop_reason="end_turn")
                        await self._queue.put(message)
                        self._status = SessinStatus.INIT
            elif msg.get("method") == "permission.request":
                await self._send_permission_response(msg["id"], msg["params"]["permissionRequest"])

    async def _send_permission_response(self, tool_call_id: str, permission_request):
        await self._client.send_result(id=tool_call_id, result={"kind": "denied-by-rules"})

    def _wait(self):
        fut = asyncio.run_coroutine_threadsafe(self._queue.get(), self._client.loop)
        return fut.result()

    def _build_athropic_message(self, block, stop_reason) -> Message:
        message = Message(id='dummy',
                          content=[block],
                          model='',
                          role='assistant',
                          type='message',
                          stop_reason=stop_reason,
                          usage={'input_tokens': 0, 'output_tokens': 0})
        return message


class CopilotSyncClient:
    def __init__(
        self,
        transport: Transport,
    ):
        self._transport = transport
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._runloop, daemon=True)

        self._id = 0
        self._queue = asyncio.Queue()
        self._sessions: Dict[str, CopilotSession] = {}  # session_id -> CopilotSession

        self._ready = threading.Event()

        # current request_id from client
        self._client_pending_id = None

        self.start()

    def start(self):
        self._thread.start()
        self._ready.wait() # <- block until connect() is done
        self.ping()

    def _runloop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main())

    async def send_message(self, method, params: Dict[str, Any]) -> str:
        rpc_id = self._next_id()
        msg = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
            "params": params
        }

        msg_str = json.dumps(msg)
        logger.debug(f"--> SEND: {msg_str}\n")

        await self._transport.write_message(msg_str)
        return rpc_id

    async def send_result(self, id, result: dict):
        msg = {
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "result": result
            },
        }

        msg_str = json.dumps(msg)
        logger.debug(f"---> SEND: {msg_str}\n")

        await self._transport.write_message(msg_str)

    async def _main(self):
        await self._transport.connect()
        self._ready.set()

        await self._eventloop()

    async def _eventloop(self):
        while True:
            line = await self._transport.read_message()
            if not line:
                break # EOF or connection closed

            logger.debug(f"<--- RECV: {line}")
            try:
                msg = json.loads(line)
            except Exception as e:
                logger.error("Failed to parse message:", e)
                continue

            await self._handle_message(msg)

    async def _handle_message(self, msg: Dict[str, Any]):
        params = msg.get("params")
        if params:
            session_id = params.get("sessionId")
            session = self._sessions.get(session_id)

            if session:
                # process session message
                await session.handle_message(msg)
        else:
            id = msg.get("id")
            if id == self._client_pending_id:
                # process client message
                result = msg.get("result")
                if result:
                    self._client_pending_id = None
                    await self._queue.put(result)

    # SYNC WAIT
    def _wait(self):
        fut = asyncio.run_coroutine_threadsafe(self._queue.get(), self.loop)
        return fut.result()

    def _next_id(self) -> str:
        self._id += 1
        return str(self._id)

    # ----------------------------
    # PUBLIC API
    # ----------------------------
    def create_session(self, model: str,
                       system_message: str | None = None,
                       tools: dict | None = None) -> CopilotSession:
        
        system_message_dict = None
        tools_dict = None

        if system_message:
            system_message_dict = {
                "content": system_message,
            }

        if tools:
            tools_dict = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    # "overridesBuiltInTool": True,
                    "parameters": {
                        "type": t["input_schema"]["type"],
                        "required": t["input_schema"]["required"],
                        "properties": t["input_schema"]["properties"]
                    }
                }
                for t in tools
            ]

        params = {
            "model": model,
            "systemMessage": system_message_dict,
            "tools": tools_dict,
            "requestPermission": True,
            "streaming": False
        }

        fut = asyncio.run_coroutine_threadsafe(
            self.send_message("session.create", params),
            self.loop
        )

        self._client_pending_id = fut.result()
        session_result = self._wait()
        session_id = session_result.get("sessionId")
        session = CopilotSession(session_id, self)
        self._sessions[session_id] = session

        return session

    def delete_session(self, session: CopilotSession):
        session_id = session.session_id
        fut = asyncio.run_coroutine_threadsafe(
            self.send_message("session.destroy", {"sessionId": session_id}),
            self.loop
        )
        self._client_pending_id = fut.result()

        result = self._wait()
        assert result.get("success") is True, f"Failed to delete session: {result}"
        del self._sessions[session_id]

    def ping(self):
        fut = asyncio.run_coroutine_threadsafe(
            self.send_message("ping", {"message": None}),
            self.loop
        )
        self._client_pending_id = fut.result()
        ping_result = self._wait()
        assert ping_result.get("message") == "pong", f"Unexpected ping response: {ping_result}"


def get_copilot_client() -> CopilotSyncClient:
    # only support stdio now
    bin_dir = Path(copilot.__file__).parent / "bin"

    if sys.platform == "win32":
        binary_name = "copilot.exe"
    else:
        binary_name = "copilot"

    copilot_cli = bin_dir / binary_name
    transport = StdioTransport(copilot_cli)

    client = CopilotSyncClient(transport)
    return client
