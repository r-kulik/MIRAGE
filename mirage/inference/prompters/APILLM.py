from typing import List, Optional, Self
import requests

from pydantic import BaseModel

from mirage import ChunkStorage

class LlmRequestParamsDTO(BaseModel):
    prompt: Optional[str]
    max_length: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    num_return_sequences: int
    

class LLM:
    def __init__(
        self, 
        max_length: int = 5000,
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1
    ) -> Self:
        
        self.url = "http://192.168.31.139:49100/generate"
        
        self.data = LlmRequestParamsDTO(
            prompt='',
            max_length=max_length, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            repetition_penalty=repetition_penalty, 
            num_return_sequences=num_return_sequences
        )
        
    def do_request(self, query: str, chunk_storage: ChunkStorage, indexes:List[str], prompt: str):
        #TODO Придумать что-то с промтом
        prompt = 'Ответь на вопрос:{query}\nЗная эту информацию:{chunks}'
        txt_chank = '\n------\n' + '\n------\n'.join([chunk_storage[i] for i in indexes]) + '\n------\n'
        txt_chank += 'endtoken'
        final = prompt.format(query=query, chunks=txt_chank)        
            
        self.data.prompt = final
        response = requests.post(self.url, json=self.data.dict())

        if response.status_code == 200:
            generated_texts = response.json().get('generated_texts', [])
            generated_texts[0] = generated_texts[0][generated_texts[0].index('endtoken')+8::]
            #logits = response.json().get('logits', [])
            generation_time = response.json().get('generation_time', [])
            print(generated_texts[0])
            #print(logits)
            print(generation_time)
        else:
            print(f"Error: {response.status_code}, {response.text}")

