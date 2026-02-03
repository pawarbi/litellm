"""
Fabric Notebook Integration Test Script

DO NOT RUN WITH PYTEST. This script is designed to be copied into a
Microsoft Fabric notebook cell for end-to-end validation.

Copy this entire file into a Fabric notebook cell and run it.

Prerequisites:
  1. Install the modified LiteLLM in the Fabric notebook:
     %pip install -e /path/to/litellm
  2. Restart the kernel after installation.
  3. Run this cell.

This script tests:
  - Provider registration
  - Chat completion (non-streaming)
  - Chat completion (streaming)
  - Reasoning model (gpt-5, non-streaming)
  - Embedding
  - Function calling
  - MwcToken auth scheme verification
  - Fabric environment detection
"""

import sys
import traceback

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

_results = []


def _run_test(name, func):
    """Run a single test and record the result."""
    try:
        result = func()
        if result is None:
            _results.append((PASS, name, ""))
        elif isinstance(result, str) and result.startswith("SKIP"):
            _results.append((SKIP, name, result))
        else:
            _results.append((PASS, name, str(result)))
    except Exception as e:
        _results.append((FAIL, name, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# Test 1: Provider registration
# --------------------------------------------------------------------------- #
def _check_provider_registration():
    import litellm
    from litellm.constants import LITELLM_CHAT_PROVIDERS
    from litellm.types.utils import LlmProviders

    errors = []

    if not hasattr(LlmProviders, "FABRIC"):
        errors.append("LlmProviders.FABRIC not found in enum")
    if "fabric" not in LITELLM_CHAT_PROVIDERS:
        errors.append("'fabric' not in LITELLM_CHAT_PROVIDERS")
    if "fabric" not in litellm.models_by_provider:
        errors.append("'fabric' not in litellm.models_by_provider")

    model, provider, _, _ = litellm.get_llm_provider("fabric/gpt-4.1")
    if provider != "fabric":
        errors.append(
            f"get_llm_provider returned provider={provider}, expected 'fabric'"
        )
    if model != "gpt-4.1":
        errors.append(f"get_llm_provider returned model={model}, expected 'gpt-4.1'")

    if errors:
        raise AssertionError("; ".join(errors))
    return "All registration checks passed"


_run_test("Provider Registration", _check_provider_registration)


# --------------------------------------------------------------------------- #
# Test 2: Chat completion (non-streaming)
# --------------------------------------------------------------------------- #
def _check_chat_completion():
    import litellm

    response = litellm.completion(
        model="fabric/gpt-4.1",
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
    )

    content = response.choices[0].message.content
    assert content is not None, "Response content is None"
    assert len(content) > 0, "Response content is empty"
    assert response.model is not None, "Response model is None"
    return f"model={response.model}, content={content[:80]}"


_run_test("Chat Completion (non-streaming)", _check_chat_completion)


# --------------------------------------------------------------------------- #
# Test 3: Chat completion (streaming)
# --------------------------------------------------------------------------- #
def _check_chat_completion_streaming():
    import litellm

    response = litellm.completion(
        model="fabric/gpt-4.1",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )

    chunks = []
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            chunks.append(delta.content)

    full_content = "".join(chunks)
    assert len(full_content) > 0, "Streaming produced no content"
    return f"chunks={len(chunks)}, content={full_content[:80]}"


_run_test("Chat Completion (streaming)", _check_chat_completion_streaming)


# --------------------------------------------------------------------------- #
# Test 4: Reasoning model (gpt-5)
# --------------------------------------------------------------------------- #
def _check_reasoning_model():
    import litellm

    try:
        response = litellm.completion(
            model="fabric/gpt-5",
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
        )
        content = response.choices[0].message.content
        assert content is not None, "Response content is None"
        return f"model={response.model}, content={content[:80]}"
    except Exception as e:
        # gpt-5 may not be available in all Fabric workspaces
        if "404" in str(e) or "not found" in str(e).lower():
            return "SKIP: gpt-5 not available in this workspace"
        raise


_run_test("Reasoning Model (gpt-5)", _check_reasoning_model)


# --------------------------------------------------------------------------- #
# Test 5: Embedding
# --------------------------------------------------------------------------- #
def _check_embedding():
    import litellm

    response = litellm.embedding(
        model="fabric/text-embedding-ada-002",
        input=["Hello world"],
    )

    assert len(response.data) > 0, "No embedding data returned"
    embedding = response.data[0]["embedding"]
    assert len(embedding) > 0, "Embedding vector is empty"
    return f"dimensions={len(embedding)}"


_run_test("Embedding", _check_embedding)


# --------------------------------------------------------------------------- #
# Test 6: Chat with function calling
# --------------------------------------------------------------------------- #
def _check_function_calling():
    import litellm

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = litellm.completion(
        model="fabric/gpt-4.1",
        messages=[{"role": "user", "content": "What is the weather in Seattle?"}],
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    has_tool_call = message.tool_calls is not None and len(message.tool_calls) > 0
    has_content = message.content is not None and len(message.content) > 0
    assert has_tool_call or has_content, "Neither tool_calls nor content in response"
    if has_tool_call:
        return f"tool_call={message.tool_calls[0].function.name}"
    return f"content={message.content[:80]}"


_run_test("Function Calling", _check_function_calling)


# --------------------------------------------------------------------------- #
# Test 7: Verify MwcToken auth scheme (not Bearer)
# --------------------------------------------------------------------------- #
def _check_auth_scheme():
    from litellm.llms.fabric.common_utils import get_fabric_auth_header

    auth_header = get_fabric_auth_header()
    assert auth_header.startswith(
        "MwcToken "
    ), f"Expected auth header to start with 'MwcToken ', got: {auth_header[:20]}..."
    return f"auth_scheme=MwcToken, token_length={len(auth_header)}"


_run_test("Auth Scheme (MwcToken)", _check_auth_scheme)


# --------------------------------------------------------------------------- #
# Test 8: Fabric environment detection
# --------------------------------------------------------------------------- #
def _check_fabric_environment():
    from litellm.llms.fabric.common_utils import is_fabric_environment

    result = is_fabric_environment()
    assert (
        result is True
    ), "is_fabric_environment() should return True in Fabric notebook"
    return "Detected Fabric environment"


_run_test("Fabric Environment Detection", _check_fabric_environment)


# --------------------------------------------------------------------------- #
# Print results summary
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("FABRIC PROVIDER INTEGRATION TEST RESULTS")
print("=" * 70)

_passed = sum(1 for r in _results if r[0] == PASS)
_failed = sum(1 for r in _results if r[0] == FAIL)
_skipped = sum(1 for r in _results if r[0] == SKIP)

for _status, _name, _detail in _results:
    print(f"  {_status} {_name}")
    if _detail:
        print(f"         {_detail}")

print("-" * 70)
print(
    f"Total: {len(_results)} | Passed: {_passed} | Failed: {_failed} | Skipped: {_skipped}"
)
print("=" * 70)

if _failed > 0:
    print("\nSome tests FAILED. Check the output above for details.")
else:
    print("\nAll tests passed. The Fabric provider is working correctly.")
