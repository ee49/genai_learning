from langchain.llms import OpenAI
import os
api_key = os.getenv('OPEN_API_KEY')
llm = OpenAI(openai_api_key=api_key)

result = llm("Write few romatic lines for my wife, who is 33 year old and loves to read books.")
print(result)
