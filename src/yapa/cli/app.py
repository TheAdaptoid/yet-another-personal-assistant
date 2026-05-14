import websockets
import asyncio
import orjson
import rich
from rich.console import Console
from yapa.shared.models import ChatRequest, ChatResponse
import requests

CHAT_URI = "ws://localhost:8000/chat/ws"
SESSION_URI = "http://localhost:8000/sessions/"


async def test_chat():
    async with websockets.connect(CHAT_URI) as websocket:
        try:
            # Send chat message
            chat_request = ChatRequest(
                session_id="test-session", model="default", message="Hello, YAPA!"
            )
            await websocket.send(orjson.dumps(chat_request.model_dump()).decode())
            chat_response = ChatResponse(**orjson.loads(await websocket.recv()))
            print("Chat response:", chat_response.response)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")


class App:
    def __init__(
        self,
        chat_uri: str = CHAT_URI,
        session_id: str | None = None,
        model: str | None = None,
    ):
        self.console = Console()
        self.chat_uri = chat_uri
        self.session_id = session_id
        self.model = model

    def create_session(self, title: str | None = None) -> str:
        response = requests.post(SESSION_URI, json={"title": title})
        if response.status_code == 201:
            session = response.json()
            self.console.print(
                f"[bold green]Created session:[/bold green] {session['id']} - {session['title']}"
            )
            return session["id"]
        else:
            self.console.print(
                f"[bold red]Failed to create session:[/bold red] {response.text}"
            )
            raise Exception("Session creation failed")

    def parse_command(self, command: str) -> None:
        if command.startswith("/session"):
            parts = command.split(maxsplit=1)
            if len(parts) == 2:
                self.session_id = parts[1]
                self.console.print(
                    f"[bold green]Session set to:[/bold green] {self.session_id}"
                )
            else:
                self.console.print("[bold red]Usage:[/bold red] /session [session_id]")
        elif command.startswith("/model"):
            parts = command.split(maxsplit=1)
            if len(parts) == 2:
                self.model = parts[1]
                self.console.print(
                    f"[bold green]Model set to:[/bold green] {self.model}"
                )
            else:
                self.console.print("[bold red]Usage:[/bold red] /model [model_name]")
        else:
            self.console.print(f"[bold red]Unknown command:[/bold red] {command}")

    async def run(
        self,
    ) -> None:
        self.console.print("[bold green]Starting YAPA CLI App...[/bold green]")

        # TODO: Implement session management and model selection
        self.session_id = "test-session"
        self.model = "default"

        async with websockets.connect(self.chat_uri) as websocket:
            try:
                while True:
                    # Get user input from the console
                    user_message: str = self.console.input(
                        "[bold blue]You:[/bold blue] "
                    )
                    if not user_message.strip():
                        continue
                    if user_message.lower() in {"exit", "quit"}:
                        self.console.print("[bold red]Exiting...[/bold red]")
                        break
                    if user_message.lower().startswith("/"):
                        self.parse_command(user_message)
                        continue

                    # Send the chat message to the server
                    chat_request = ChatRequest(
                        session_id=self.session_id,
                        model=self.model,
                        message=user_message,
                    )
                    await websocket.send(chat_request.model_dump_json())

                    # Wait for the response from the server
                    chat_response = ChatResponse.model_validate_json(
                        await websocket.recv()
                    )
                    if chat_response.error:
                        self.console.print(
                            f"[bold red]Error from server:[/bold red] {chat_response.error}"
                        )
                    else:
                        self.console.print(
                            f"[bold green]YAPA:[/bold green] {chat_response.response}"
                        )

                    while not chat_response.done:
                        chat_response = ChatResponse.model_validate_json(
                            await websocket.recv()
                        )
                        if chat_response.error:
                            self.console.print(
                                f"[bold red]Error from server:[/bold red] {chat_response.error}"
                            )
                            break
                        else:
                            self.console.print(
                                f"[bold green]YAPA:[/bold green] {chat_response.response}"
                            )

                await websocket.close()
            except websockets.exceptions.ConnectionClosed:
                self.console.print("[bold red]WebSocket connection closed[/bold red]")


if __name__ == "__main__":
    app = App()
    asyncio.run(app.run())
