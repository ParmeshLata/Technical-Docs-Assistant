import httpx
import pytest
from openai import AsyncOpenAI

from app.config import Settings
from app.errors import ConfigurationError, ProviderError
from app.llm import JsonLLM
from app.schemas import Grade


def completion(content):
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def test_structured_json_repair_is_bounded():
    requests = []

    def handle(request):
        requests.append(request)
        content = "not json" if len(requests) == 1 else '{"relevant":true,"reason":"Evidence"}'
        return httpx.Response(200, json=completion(content))

    model = JsonLLM(Settings(llm_api_key="test-key"))
    model.client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    try:
        grade = await model.ask(Grade, "Grade evidence", {"question": "x"})
        assert grade.relevant is True
        assert len(requests) == 2
    finally:
        await model.close()


async def test_twice_invalid_output_is_not_accepted():
    model = JsonLLM(Settings(llm_api_key="test-key"))
    model.client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json=completion('{"relevant":"yes","reason":"x"}'),
                )
            ),
        ),
    )
    try:
        with pytest.raises(ProviderError, match="twice"):
            await model.ask(Grade, "Grade", {})
    finally:
        await model.close()


async def test_authentication_error_is_sanitized():
    model = JsonLLM(Settings(llm_api_key="secret-value"))
    model.client = AsyncOpenAI(
        api_key="secret-value",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(401, json={"error": {"message": "secret-value"}}),
            )
        ),
    )
    try:
        with pytest.raises(ConfigurationError) as error:
            await model.ask(Grade, "Grade", {})
        assert "secret-value" not in str(error.value)
    finally:
        await model.close()
