import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI, AsyncOpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class LLM:
    """
    A class for interacting with OpenAI's GPT models via the API.
    """

    DEFAULT_SYSTEM_PROMPT = "translate the following text to Persian, not answer it just translate ir."

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ):
        """
        Initializes the LLM object.

        Args:
            api_key (str, optional): Your OpenAI API key. If None, it will attempt to read it from the `OPENAI_API_KEY` environment variable.
            model (str): The name of the GPT model to use (e.g., "gpt-4", "gpt-3.5-turbo").
            temperature (float): Controls randomness (0.0 = deterministic, 1.0 = highly random).
            max_tokens (int): The maximum number of tokens to generate in the response.

        Raises:
            ValueError: If the API key is not provided or cannot be found in the environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key provided. Set the OPENAI_API_KEY environment variable or pass it directly."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}
        ]

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the conversation history.

        Args:
            role (str): The role of the message sender (e.g., "system", "user", "assistant").
            content (str): The content of the message.

        Raises:
            ValueError: If an invalid role is provided.
        """
        if role not in ["system", "user", "assistant"]:
            raise ValueError("Invalid role. Must be 'system', 'user', or 'assistant'.")
        self.messages.append({"role": role, "content": content})

    def clear_messages(self) -> None:
        """
        Clears the conversation history and resets to the default system prompt.
        """
        self.messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]

    def generate_response(self, prompt: Optional[str] = None) -> str:
        """
        Generates a response from the GPT model.

        Args:
            prompt (str, optional): The text prompt to send to the model. If messages are present, this is appended as a user message.

        Returns:
            str: The generated response as a string.

        Raises:
            RuntimeError: If there is an error with the OpenAI API.
        """
        if prompt:
            self.add_message("user", prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            raise RuntimeError("Failed to generate response from OpenAI API.") from e

    async def async_generate_response(self, prompt: Optional[str] = None) -> str:
        """
        Generates a response from the GPT model asynchronously.

        Args:
            prompt (str, optional): The text prompt to send to the model. If messages are present, this is appended as a user message.

        Returns:
            str: The generated response as a string.

        Raises:
            RuntimeError: If there is an error with the OpenAI API.
        """
        if prompt:
            self.add_message("user", prompt)

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API Error: {e}")
            raise RuntimeError("Failed to generate response from OpenAI API.") from e

    def __str__(self) -> str:
        """
        Returns a string representation of the LLM object.

        Returns:
            str: String representation of the object, including model and main parameters.
        """
        return f"LLM(model='{self.model}', temperature={self.temperature}, max_tokens={self.max_tokens})"