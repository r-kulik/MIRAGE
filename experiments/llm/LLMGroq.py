from typing import List, Optional
from groq import Groq
from loguru import logger
from pydantic import BaseModel

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

class GroqLLM():
    def __init__(
        self, 
        model: str = 'llama3-8b-8192',  # Default model
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
        frequency_penalty: float = 1.2,
        n: int = 1,
        api_key_num: Optional[int] = 0
    ) -> None:
        
        # Initialize the Groq client
        self.client = Groq(api_key=groq_api_keys[api_key_num] )
        
        # Initialize request parameters
        self.data = GroqRequestParamsDTO(
            messages=[],  # Messages will be added dynamically
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            n=n
        )
        
    def do_request(self, query: str, chunks: List[str], prompt_template: Optional[str] = None):
        # Default prompt template
        if prompt_template is None:
            prompt_template = 'Answer the question: {query}\nKnowing this information: {chunks}'
        
        # Construct the chunks string
        txt_chunk = '\n------\n' + '\n------\n'.join(chunks) + '\n------\n'
        
        # Format the final prompt
        final_prompt = prompt_template.format(query=query, chunks=txt_chunk)   
        return final_prompt     
        # Prepare the messages for the Groq API
        self.data.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_prompt}
        ]
        
        # Make the request using the Groq client
        response = self.client.chat.completions.create(
            messages=self.data.messages,
            model=self.data.model,
            max_tokens=self.data.max_tokens,
            temperature=self.data.temperature,
            top_p=self.data.top_p,
            frequency_penalty=self.data.frequency_penalty,
            n=self.data.n
        )
        
        # Process the response
        if response.choices:
            generated_text = response.choices[0].message.content
            print(generated_text)
            return generated_text
        else:
            error_text = "No completions found in the response."
            logger.error(error_text)
            raise ValueError(error_text)
