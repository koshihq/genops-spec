"""Mock provider implementations for testing."""

import time
from typing import Optional
from unittest.mock import DEFAULT, MagicMock


class MockOpenAIResponse:
    """Mock OpenAI API response."""

    def __init__(
        self,
        content: str = "Test response",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        model: str = "gpt-3.5-turbo",
    ):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content

        self.usage = MagicMock()
        self.usage.prompt_tokens = prompt_tokens
        self.usage.completion_tokens = completion_tokens
        self.usage.total_tokens = prompt_tokens + completion_tokens

        self.model = model
        self.id = f"chatcmpl-test-{int(time.time())}"
        self.object = "chat.completion"
        self.created = int(time.time())


class MockAnthropicResponse:
    """Mock Anthropic API response."""

    def __init__(
        self,
        content: str = "Test Claude response",
        input_tokens: int = 12,
        output_tokens: int = 8,
        model: str = "claude-3-sonnet-20240229",
    ):
        self.content = [MagicMock()]
        self.content[0].text = content
        self.content[0].type = "text"

        self.usage = MagicMock()
        self.usage.input_tokens = input_tokens
        self.usage.output_tokens = output_tokens

        self.model = model
        self.id = f"msg_test_{int(time.time())}"
        self.type = "message"
        self.role = "assistant"


class MockProviderFactory:
    """Factory for creating mock provider responses."""

    # OpenAI model pricing (cost per 1K tokens)
    OPENAI_PRICING = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }

    # Anthropic model pricing (cost per 1K tokens)
    ANTHROPIC_PRICING = {
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-instant-1.2": {"input": 0.00163, "output": 0.00551},
    }

    @staticmethod
    def create_openai_response(
        model: str = "gpt-3.5-turbo",
        content: str = "Test AI response",
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> MockOpenAIResponse:
        """Create a mock OpenAI response with realistic token counts."""
        # Estimate tokens if not provided
        if prompt_tokens is None:
            prompt_tokens = max(10, len(content.split()) // 2)
        if completion_tokens is None:
            completion_tokens = max(5, len(content.split()))

        return MockOpenAIResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )

    @staticmethod
    def create_anthropic_response(
        model: str = "claude-3-sonnet-20240229",
        content: str = "Test Claude response",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> MockAnthropicResponse:
        """Create a mock Anthropic response with realistic token counts."""
        # Estimate tokens if not provided
        if input_tokens is None:
            input_tokens = max(12, len(content.split()) // 2)
        if output_tokens is None:
            output_tokens = max(8, len(content.split()))

        return MockAnthropicResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

    @staticmethod
    def calculate_openai_cost(
        model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate cost for OpenAI request."""
        pricing = MockProviderFactory.OPENAI_PRICING.get(
            model, {"input": 0.0005, "output": 0.0015}
        )

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)

    @staticmethod
    def calculate_anthropic_cost(
        model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for Anthropic request."""
        pricing = MockProviderFactory.ANTHROPIC_PRICING.get(
            model, {"input": 0.003, "output": 0.015}
        )

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return round(input_cost + output_cost, 6)


class MockProviderClient:
    """Base mock provider client."""

    def __init__(self, fail_requests: bool = False, delay: float = 0.1):
        self.fail_requests = fail_requests
        self.delay = delay
        self.request_count = 0

    def _simulate_delay(self):
        """Simulate network delay."""
        if self.delay > 0:
            time.sleep(self.delay)

    def _check_failure(self):
        """Check if request should fail."""
        if self.fail_requests:
            raise Exception("Mock API error")
        self.request_count += 1


class MockOpenAIClient(MockProviderClient):
    """Mock OpenAI client for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat = MagicMock()
        self.chat.completions = MagicMock()

        # Set up chat completions create method with proper mock behavior
        self.chat.completions.create = MagicMock(
            side_effect=self._chat_completions_create
        )

        # Add completions for legacy completions API
        self.completions = MagicMock()
        self.completions.create = MagicMock(side_effect=self._completions_create)

    def reset_mock(self):
        """Reset all mock call counts and side effects."""
        self.request_count = 0
        self.chat.completions.create.reset_mock()
        self.completions.create.reset_mock()

    def _default_chat_response(self) -> MockOpenAIResponse:
        """Create default response for when mock is accessed without parameters."""
        return MockProviderFactory.create_openai_response()

    def _chat_completions_create(self, **kwargs) -> MockOpenAIResponse:
        """Mock chat completions create method."""
        self._simulate_delay()
        self._check_failure()

        model = kwargs.get("model", "gpt-3.5-turbo")
        messages = kwargs.get("messages", [])

        # Estimate content length for response
        input_text = " ".join([msg.get("content", "") for msg in messages])
        response_content = f"AI response to: {input_text[:50]}..."

        return MockProviderFactory.create_openai_response(
            model=model, content=response_content
        )

    def _completions_create(self, **kwargs) -> MockOpenAIResponse:
        """Mock completions create method (legacy API)."""
        self._simulate_delay()
        self._check_failure()

        model = kwargs.get("model", "gpt-3.5-turbo")
        prompt = kwargs.get("prompt", "")

        # Generate response content
        response_content = f"AI response to: {prompt[:50]}..."

        return MockProviderFactory.create_openai_response(
            model=model, content=response_content
        )


class MockAnthropicClient(MockProviderClient):
    """Mock Anthropic client for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = MagicMock()

        # Use MagicMock with side_effect so tests can override via return_value.
        # The wrapper returns DEFAULT when return_value has been explicitly set,
        # which tells MagicMock to use return_value instead.
        self.messages.create = MagicMock(side_effect=self._messages_create_wrapper)
        self._default_return_value = self.messages.create.return_value

    def reset_mock(self):
        """Reset all mock call counts and side effects."""
        self.request_count = 0
        self.messages.create.reset_mock(side_effect=self._messages_create_wrapper)
        self._default_return_value = self.messages.create.return_value

    def _messages_create_wrapper(self, **kwargs) -> MockAnthropicResponse:
        """Wrapper that defers to return_value when it has been overridden."""
        # If a test set messages.create.return_value, honour it.
        # Skip side-effects (delay/failure) since the test controls the response.
        if self.messages.create.return_value is not self._default_return_value:
            return DEFAULT
        return self._messages_create(**kwargs)

    def _messages_create(self, **kwargs) -> MockAnthropicResponse:
        """Mock messages create method."""
        self._simulate_delay()
        self._check_failure()

        model = kwargs.get("model", "claude-3-sonnet-20240229")
        messages = kwargs.get("messages", [])

        # Estimate content length for response
        input_text = " ".join([msg.get("content", "") for msg in messages])
        response_content = f"Claude response to: {input_text[:50]}..."

        return MockProviderFactory.create_anthropic_response(
            model=model, content=response_content
        )
