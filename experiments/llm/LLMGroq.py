from typing import List, Literal, Optional
from groq import Groq
from loguru import logger
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from groq_keys import groq_api_keys
import os


class GroqRequestParamsDTO(BaseModel):
    messages: List[dict]  # List of messages for the chat completion
    model: str  # Model to use (e.g., "llama-3.3-70b-versatile")
    max_tokens: Optional[int] = 1024  # Maximum number of tokens to generate
    temperature: Optional[float] = 0.2  # Sampling temperature
    top_p: Optional[float] = 0.9  # Nucleus sampling parameter
    frequency_penalty: Optional[float] = 1.2  # Frequency penalty
    n: Optional[int] = 1  # Number of completions to generate


class GroqLLM:
    def __init__(
        self,
        model: str = "llama3-8b-8192",  # Default model
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
        frequency_penalty: float = 1.2,
        n: int = 1,
        api_key_num: Optional[int] = 0,
    ) -> None:

        # Initialize the Groq client
        self.client = Groq(api_key=groq_api_keys[api_key_num])

        # Initialize request parameters
        self.data = GroqRequestParamsDTO(
            messages=[],  # Messages will be added dynamically
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            n=n,
        )

        self.json_llm = ChatGroq(
            api_key=groq_api_keys[api_key_num],
            model="qwen-qwq-32b",
            temperature=0.6,
            top_p=0.95,
            frequency_penalty=2,
            reasoning_format="parsed",
        )

        self.json_parser = JsonOutputParser(
            pydantic_object={
                "type": "object",
                "properties": {
                    "correctness": {"type": "integer", "enum": [0, 1]},
                    "completeness": {"type": "integer", "minimum": 0, "maximum": 10},
                },
            }
        )

        self.ru_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Прочитай вопрос, идеальный ответ на этот вопрос и ответ, который дала модель. Оцени следующие параметры:
                    {{
                        "correctness": 1, если ответ модели правильный, 0 если неправильный
                        "completness": целое число от 0 до 10 включительно, где 5 - полное совпадение данного ответа с идеальным, 0 - в данном моделью ответе настолько недостаточно информации, что нет ответа на поставленный вопрос, 10 - переизбыток информации в данном моделью ответе привел к ответу на незаданные пользователем вопросы.
                    }}
                 Ответ верни в формате JSON, приведенном выше""",
                ),
                (
                    "user",
                    """Вопрос пользователя: {user_question}\n----------\nИдеальный ответ: {ideal_answer}\n----------\nОтвет модели: {llm_answer}""",
                ),
            ]
        )

        self.en_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Read question, ideal answer to this question and an answer that was provided by LLM. Prrovide values for following parameters:
                    {{
                        "correctness": 1 is model answer is correct 0 otherwise
                        "completness": integer from 0 to 10 inclusively, where 5 - answers are fully analagous. 0 - in the model response there is a luck of information such big that there is no answer to the pointed question, 10 - overflow of information leads to occurence of additional answers to unpointed questions.
                    }}
                 Respond in the stated JSON format""",
                ),
                (
                    "user",
                    """User question: {user_question}\n----------\nIdeal answer: {ideal_answer}\n----------\nModel answer: {llm_answer}""",
                ),
            ]
        )

    def do_request(
        self,
        query: str,
        chunks: List[str],
        lang: Literal["ru", "en"],
        prompt_template: Optional[str] = None,
    ):

        # Default prompt template
        if prompt_template is None:
            match lang:
                case "ru":
                    system_prompt = "Ты - ассистент юридической поддержки. Отвечай на запросы точно на основании предоставленных документов. Если ответа на вопрос нет в прочитанных документах, отвечай, что не знаешь ответа на вопрос."
                    prompt_template = (
                        "Вопрос: {query}\n\nФрагменты документов:\n{chunks}"
                    )
                case "en":
                    system_prompt = "You are an juridical assistant. Answer the questions strictly based on provided documents. If an answer to the question is not presented in the text, respond that you do not know the answer"
                    prompt_template = "Question: {query}\n\nDocuments:\n{chunks}"
                case _:
                    raise ValueError("unsupported language")

        # Construct the chunks string
        txt_chunk = "\n------\n" + "\n------\n".join(chunks) + "\n------\n"

        # Format the final prompt
        final_prompt = prompt_template.format(query=query, chunks=txt_chunk)
        # logger.warning(final_prompt)
        # return final_prompt
        # Prepare the messages for the Groq API
        self.data.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]

        # Make the request using the Groq client
        response = self.client.chat.completions.create(
            messages=self.data.messages,
            model=self.data.model,
            max_tokens=self.data.max_tokens,
            temperature=self.data.temperature,
            top_p=self.data.top_p,
            frequency_penalty=self.data.frequency_penalty,
            n=self.data.n,
        )

        # Process the response
        if response.choices:
            generated_text = response.choices[0].message.content
            return generated_text
        else:
            error_text = "No completions found in the response."
            logger.error(error_text)
            raise ValueError(error_text)

    def do_request_base(
        self,
        query: str,
        lang: Literal["ru", "en"],
        prompt_template: Optional[str] = None,
    ):

        # Default prompt template
        if prompt_template is None:
            match lang:
                case "ru":
                    system_prompt = "Ты - ассистент юридической поддержки. Отвечай на запросы точно."
                    prompt_template = "Вопрос: {query}"
                case "en":
                    system_prompt = (
                        "You are an juridical assistant. Answer the questions strictly."
                    )
                    prompt_template = "Question: {query}"
                case _:
                    raise ValueError("unsupported language")

        # Format the final prompt
        final_prompt = prompt_template.format(query=query)
        # return final_prompt
        # Prepare the messages for the Groq API
        self.data.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]

        # Make the request using the Groq client
        response = self.client.chat.completions.create(
            messages=self.data.messages,
            model=self.data.model,
            max_tokens=self.data.max_tokens,
            temperature=self.data.temperature,
            top_p=self.data.top_p,
            frequency_penalty=self.data.frequency_penalty,
            n=self.data.n,
        )

        # Process the response
        if response.choices:
            generated_text = response.choices[0].message.content
            return generated_text
        else:
            error_text = "No completions found in the response."
            logger.error(error_text)
            raise ValueError(error_text)

    def reveal_correctness(
        self,
        question: str,
        ideal_answer: str,
        llm_answer: str,
        lang=Literal["ru", "en"],
    ) -> dict:
        match lang:
            case "ru":
                chain = self.ru_prompt_template | self.json_llm | self.json_parser
            case "en":
                chain = self.en_prompt_template | self.json_llm | self.json_parser
            case _:
                logger.error("у тя че за язык")
                raise ValueError("unknown language")
        result = chain.invoke(
            {
                "user_question": question,
                "ideal_answer": ideal_answer,
                "llm_answer": llm_answer,
            },
        )
        return result
