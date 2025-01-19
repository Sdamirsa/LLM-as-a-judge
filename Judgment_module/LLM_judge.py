import sys
import os

# Add the parent directory (where Judgment_module is located) to Python path
module_path = os.path.abspath(os.path.join(os.getcwd(), 'Judgment_module'))
if module_path not in sys.path:
    sys.path.append(module_path)

# general imports
from typing import List, Union,Optional,Any
from enum import IntEnum,Enum
from pydantic import BaseModel, Field # type: ignore
import logging
import asyncio
import json

# local imports
from Judgment_module.pydantic_classes import LLMJudgeBool,LLMJudgeFit,LLMJudgeOneFour,LLMJudgeBorderline,LLMJudgeLikert
from Judgment_module.LLM_package import LLM

###########################################################################################################
# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


####################################################################################################



class JudgmentLLM(LLM):
    """
    A subclass of LLM for performing specific judgment tasks.
    """
    ALLOWED_OPTIONS = ["ground_truth_judgment", "discrepancy_judgment", "fact_check_judgment"]
    ALLOWED_GRADING_TYPES = ["boolean", "1_to_4", "bad_good_best_fit","pass_borderline_fail","likert_scale_1_to_5"]

    PROMPTS = {}
    GRADING_PROMPTS = {}
    RESPONSE_MODEL_MAPPING = {
    "boolean": LLMJudgeBool,
    "1_to_4": LLMJudgeOneFour,
    "bad_good_best_fit": LLMJudgeFit,
    "pass_borderline_fail": LLMJudgeBorderline,
    "likert_scale_1_to_5": LLMJudgeLikert
}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        option: str = None,
        user_prompt: bool = False,
        grading_type: str = None):
        """
        Initializes the JudgmentLLM object.

        Args:
            api_key (str, optional): Your OpenAI API key.
            model (str): The name of the GPT model to use.
            temperature (float): Controls response randomness.
            max_tokens (int): The maximum number of tokens in the response.
            option (str): Required judgment type, must be one of: "boolean_judgment", "discrepancy_judgment", "fact_check_judgment".
        Raises:
            ValueError: If the 'option' is invalid.
        """
        super().__init__(api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens)
        self.load_prompts()
        self.option = self._validate_option(option)
        
        self.user_prompt = user_prompt
        self.grading_type = self._validate_grading_type(grading_type)
        self.grading_prompt = self.GRADING_PROMPTS[self.grading_type]

        if not user_prompt:
            self.system_prompt = self.PROMPTS[self.option]
        else:
            self.system_prompt = self.PROMPTS[f'{self.option}_with_user_prompt']
        self.response_model = self._get_response_model(self.grading_type)
        

    @staticmethod
    def _validate_option(option: str) -> str:
        """Validates if the provided option is allowed."""
        if option is None:
            raise ValueError(f"please specify option. Allowed options are: {JudgmentLLM.ALLOWED_OPTIONS}",)
        elif option not in JudgmentLLM.ALLOWED_OPTIONS:
            raise ValueError(
                f"Invalid option '{option}'. Allowed options are: {JudgmentLLM.ALLOWED_OPTIONS}"
            )
        return option
    def _validate_grading_type(self, grading_type: str) -> str:
        """Validates if the provided grading type is allowed."""
        if grading_type is None:
            raise ValueError(f"please specify grading_type. Allowed grading_type are: {JudgmentLLM.ALLOWED_GRADING_TYPES}",)
        elif grading_type not in JudgmentLLM.ALLOWED_GRADING_TYPES:
            raise ValueError(f"Invalid grading type '{grading_type}'. Allowed grading types are: {JudgmentLLM.ALLOWED_GRADING_TYPES}")
        return grading_type

    @staticmethod
    def _get_response_model(grading_type: str) -> BaseModel:
        return JudgmentLLM.RESPONSE_MODEL_MAPPING[grading_type]
    def generate_judgment(self, text: List[str], system_prompt: Optional[str] = None) -> Any:
        """
        Generates a judgment based on the provided text and optional system prompt.
        If selected option are "boolean_judgment", "discrepancy_judgment", "fact_check_judgment" please provide list containing one string.
        If selected option is "boolean_judgment_with_user_prompt" please provide list containing two strings first user input second system response.
        
        Args:
            text (List[str]): A list of strings containing the text to be judged.
                - For boolean_judgment, discrepancy_judgment, fact_check_judgment: provide list with one string
                - For boolean_judgment_with_user_prompt: provide list with two strings [user_input, system_response]
            system_prompt (Optional[str], optional): An optional system prompt to guide the judgment generation. Defaults to None.

        Returns:
            Any: The generated judgment response.

        Raises:
            ValueError: If the input text list doesn't match the required format for the selected judgment type.
        """
        # Input validation
        if not isinstance(text, list) or not all(isinstance(item, str) for item in text):
            raise ValueError("text must be a list of strings")

        # Use the system prompt from class initialization if none provided
        if system_prompt is None:
            system_prompt = self.system_prompt
        user_prompt = self.generate_user_prompt(text)
        # Generate and return response
        res = self.generate_response(user_prompt, system_prompt)
        return res
    def generate_response(self, user_input: str, system_prompt: Optional[str] = None):
        """
        Performs the judgment task by interacting with the GPT model.

        Args:
            user_input (str): The user's input statement(s).
            system_prompt (str, optional): Custom instructions for the GPT model. 
                                           Defaults to the backup prompt.

        Returns:
            BaseModel: Parsed response in the form of the specified response model.
        Raises:
            Exception: If interaction with the GPT model fails.
        """
        prompt = system_prompt or self.system_prompt
        prompt += f"\n \n {self.GRADING_PROMPTS}"
        self.clear_messages()
        self.add_message("system", prompt)
        self.add_message("user", user_input)

        try:
            # Adjust the API call to your library/environment
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=self.messages,
                response_format=self.response_model,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Error during judgment task: {e}")
            raise RuntimeError("Failed to generate a response from the LLM.") from e

    async def generate_judgment_async(self, text: List[str], system_prompt: Optional[str] = None) -> Any:
        """
        Asynchronously generates a judgment based on the provided text and optional system prompt.
        If selected option are "boolean_judgment", "discrepancy_judgment", "fact_check_judgment" please provide list containing one string.
        If selected option is "boolean_judgment_with_user_prompt" please provide list containing two strings first user input second system response.

        Args:
            text (List[str]): A list of strings containing the text to be judged.
                - For boolean_judgment, discrepancy_judgment, fact_check_judgment: provide list with one string
                - For boolean_judgment_with_user_prompt: provide list with two strings [user_input, system_response]
            system_prompt (Optional[str], optional): An optional system prompt to guide the judgment generation. Defaults to None.

        Returns:
            Any: The generated judgment response.

        Raises:
            ValueError: If the input text list doesn't match the required format for the selected judgment type.
        """
        # Input validation
        if not isinstance(text, list) or not all(isinstance(item, str) for item in text):
            raise ValueError("text must be a list of strings")

        # Use the system prompt from class initialization if none provided
        if system_prompt is None:
            system_prompt = self.system_prompt

        user_prompt = self.generate_user_prompt(text)

        # Generate and return response asynchronously
        res = await self.generate_response_async(user_prompt, system_prompt)
        return res


    async def generate_response_async(self, user_input: str, system_prompt: Optional[str] = None):
        """
        Performs the judgment task asynchronously by interacting with the GPT model.

        Args:
            user_input (str): The user's input statement(s).
            system_prompt (str, optional): Custom instructions for the GPT model. 
                                        Defaults to the backup prompt.

        Returns:
            BaseModel: Parsed response in the form of the specified response model.
        Raises:
            Exception: If interaction with the GPT model fails.
        """
        prompt = system_prompt or self.system_prompt
        prompt += f"\n \n {self.GRADING_PROMPTS}"
        self.clear_messages()
        self.add_message("system", prompt)
        self.add_message("user", user_input)

        try:
            # Adjust the async API call to your library/environment
            response = await self.async_client.beta.chat.completions.parse(
                model=self.model,
                messages=self.messages,
                response_format=self.response_model,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Error during judgment task: {e}")
            raise RuntimeError("Failed to generate a response from the LLM.") from e
    def load_prompts(self):
        with open("prompts.json", "r") as f:
            self.PROMPTS = json.load(f)
        with open("gradings.json", "r") as k:
            self.GRADING_PROMPTS = json.load(k)
            
    def generate_user_prompt(self, text: List[str]) -> str:
        """
        Generates the appropriate user prompt based on the current judgment option and user input.

        Args:
            text (List[str]): A list of strings containing the text to be used for the user prompt.
            
        Returns:
            str: The generated user prompt.

        Raises:
            ValueError: If the input text list doesn't match the required format for the selected judgment type.
        """
        if self.user_prompt:
            if self.option == "discrepancy_judgment":
                if len(text) != 3:
                    raise ValueError(
                        "this option requires exactly three strings: [user_1_input, user_2_input, argument]"
                    )
                return f'''the general statement or question is: \n \n {text[2]}\n \n
                the text of User1 is: \n \n {text[0]}\n \n
                the text of User2 is: \n \n {text[1]}'''
            else:
                if len(text) != 2:
                    raise ValueError(
                        "this option requires exactly two strings: [general_statement, user_input]"
                    )
                return f'''the general statement or question is: \n \n {text[0]}
                the text of User input is: \n \n {text[1]}'''
        else:
            if self.option == "discrepancy_judgment":
                if len(text) != 2:
                    raise ValueError(
                        f"this {self.option} requires exactly two strings: [statement_1, statement_2]"
                    )
                return f'''the text of User1 input is: \n \n {text[0]}
                the text of User2 is: \n \n {text[1]}'''
            if self.option == "ground_truth_judgment":
                if len(text) != 2:
                    raise ValueError(
                        f"this {self.option} requires exactly two strings: [statement_1, ground_truth]"
                    )
                return f'''the answer of User is: \n \n {text[0]} \n \n
                the Ground truth is: \n \n {text[1]}'''
            else:
                if len(text) != 1:
                    raise ValueError(
                        f"{self.option} requires exactly one string in the text list"
                    )
                return text[0]
            


    def display_prompt_format(self) -> None:
        """
        Displays the current system prompt and expected user prompt format for debugging purposes.
        This helps users understand how to structure their input and what instructions the LLM is following.
        """
        # Display the system prompt
        print("=== Current System Prompt ===")
        print(self.system_prompt)
        print("\n=== Grading Type ===")
        print(f"Using grading type: {self.grading_type}")
        print(f"Grading prompt: {self.grading_prompt}")
        print("\n=== Expected User Prompt Format ===")
        
        # Display the expected format based on the current option and user_prompt setting
        print("Required input format:")
        if self.user_prompt:
            if self.option == "discrepancy_judgment":
                print("List of 3 strings: [user_1_input, user_2_input, argument]")
                print("Example usage: judge.generate_judgment(['User1 statement', 'User2 statement', 'General argument'])")
            else:
                print("List of 2 strings: [general_statement, user_input]")
                print("Example usage: judge.generate_judgment(['General statement', 'User input'])")
        else:
            if self.option == "discrepancy_judgment":
                print("List of 2 strings: [statement_1, statement_2]")
                print("Example usage: judge.generate_judgment(['Statement 1', 'Statement 2'])")
            elif self.option == "ground_truth_judgment":
                print("List of 2 strings: [statement_1, ground_truth]")
                print("Example usage: judge.generate_judgment(['User answer', 'Ground truth'])")
            else:
                print("List of 1 string: [statement]")
                print("Example usage: judge.generate_judgment(['Statement to evaluate'])")
        
        print("\n=== Response Format ===")
        print(f"Response will be parsed according to: {self.response_model.__name__}")