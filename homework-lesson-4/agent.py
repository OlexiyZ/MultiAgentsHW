from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib import error, request

from config import OPENAI_API_URL, Settings, SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY, format_tool_result


def _clip_for_log(value: str, limit: int = 400) -> str:
    """Shortens long log strings for cleaner console output.
    Скорочує довгі рядки логів для охайнішого виводу в консоль."""
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 3, 0)].rstrip() + "..."


@dataclass
class ResearchAgent:
    """Implements a stateful research agent with tool-calling over the OpenAI API.
    Реалізує дослідницького агента зі збереженням стану та викликом інструментів через OpenAI API."""
    settings: Settings
    messages: list[dict[str, Any]] = field(
        default_factory=lambda: [{"role": "system", "content": SYSTEM_PROMPT}],
    )

    def invoke(self, user_input: str) -> str:
        """Processes one user request and returns the final text answer.
        Обробляє один користувацький запит і повертає фінальний текст відповіді."""
        self.messages.append({"role": "user", "content": user_input})

        for iteration in range(1, self.settings.max_iterations + 1):
            response_message = self._request_completion()
            self.messages.append(self._build_assistant_message(response_message))

            tool_calls = response_message.get("tool_calls") or []
            if tool_calls:
                print(f"\n[Iteration {iteration}] Assistant requested {len(tool_calls)} tool(s)")
                self._execute_tool_calls(tool_calls)
                continue

            content = self._extract_text_content(response_message)
            if content:
                return content

            return "Agent finished without a textual response."

        fallback = (
            "Agent stopped because it reached the iteration limit before producing "
            "a final answer."
        )
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback

    def _request_completion(self) -> dict[str, Any]:
        """Sends the current conversation to the model and returns its raw message.
        Надсилає поточну розмову в модель і повертає її сире повідомлення."""
        payload = {
            "model": self.settings.model_name,
            "messages": self.messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto", # Треба буде спробувати примусово обирати тули за умовою.
            "temperature": self.settings.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        http_request = request.Request(
            OPENAI_API_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.settings.request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI API response: {data}") from exc

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Executes tool calls, logs results, and appends tool messages to history.
        Виконує виклики інструментів, логує результати та додає повідомлення інструментів в історію."""
        for tool_call in tool_calls:
            function_call = tool_call.get("function") or {}
            tool_name = function_call.get("name", "")
            raw_arguments = function_call.get("arguments", "{}")
            tool_id = tool_call.get("id")

            print(f"  Tool call: {tool_name}({raw_arguments})")

            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                result = {"error": f"Invalid JSON arguments for tool {tool_name}: {exc}"}
            else:
                result = self._run_tool(tool_name, arguments)

            rendered = format_tool_result(result, self.settings.max_tool_result_length)
            print(f"  Result: {_clip_for_log(rendered)}")

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": rendered,
                },
            )

    def _run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Dispatches a tool by name and converts failures into structured errors.
        Викликає інструмент за назвою та перетворює помилки у структурований результат."""
        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return tool_fn(**arguments)
        except TypeError as exc:
            return {"error": f"Invalid arguments for {tool_name}: {exc}"}
        except Exception as exc:  # pragma: no cover - defensive path
            return {"error": f"Tool {tool_name} failed: {exc}"}

    @staticmethod
    def _build_assistant_message(response_message: dict[str, Any]) -> dict[str, Any]:
        """Builds a normalized assistant message for stored conversation state.
        Формує нормалізоване повідомлення асистента для збереження стану розмови."""
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": ResearchAgent._extract_text_content(response_message),
        }
        tool_calls = response_message.get("tool_calls")
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        return assistant_message

    @staticmethod
    def _extract_text_content(response_message: dict[str, Any]) -> str:
        """Extracts plain text from the model response payload.
        Витягує звичайний текст із payload відповіді моделі."""
        content = response_message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(part for part in parts if part).strip()
        return ""


settings = Settings()
agent = ResearchAgent(settings=settings)
