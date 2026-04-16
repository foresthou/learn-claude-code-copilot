import asyncio


class Transport:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer


    async def connect(self):
        ...

    async def read_message(self) -> str:
        # Read header line
        header_line = await self.reader.readline()
        if not header_line:
            return None

        # Parse Content-Length
        header = header_line.decode("utf-8").strip()
        if not header.startswith("Content-Length:"):
            return None

        content_length = int(header.split(":")[1].strip())

        # Read empty line
        await self.reader.readline()

        # Read exact content using loop to handle short reads
        content_bytes = await self._read_exact(content_length)
        content = content_bytes.decode("utf-8")
        return content

    async def _read_exact(self, num_bytes: int) -> bytes:
        chunks = []
        remaining = num_bytes
        while remaining > 0:
            chunk = await self.reader.read(remaining)
            if not chunk:
                raise EOFError("Unexpected end of stream while reading JSON-RPC message")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    async def write_message(self, content: str):
        content_bytes = content.encode("utf-8")
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"

        self.writer.write(header.encode("utf-8"))
        self.writer.write(content_bytes)
        await self.writer.drain()


class StdioTransport(Transport):
    def __init__(self, sever_path: str):
        self._cmd = [
            sever_path, "--stdio", "--headless",
            "--no-auto-update",
            "--excluded-tools=bash,read,create,edit",
            "--deny-tool=bash",
            "--deny-tool=read",
            "--deny-tool=create",
            "--deny-tool=edit",
        ]

    async def connect(self):
        proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            bufsize=0,
        )

        self.writer = proc.stdin
        self.reader = proc.stdout


class TCPTransport:
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(
            self._host, self._port
        )
