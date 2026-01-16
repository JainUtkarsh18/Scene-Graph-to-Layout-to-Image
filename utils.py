
from jinja2 import Environment, StrictUndefined
import yaml
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json


load_dotenv()

def get_prompt(template_path, role, **kwargs):
    with open(template_path, 'r') as file:
        prompt_templates = yaml.safe_load(file)
        prompt_role = prompt_templates[role]
        environment = Environment(undefined=StrictUndefined, autoescape=False)
        jinja_template = environment.from_string(prompt_role)
        return jinja_template.render(**kwargs)
    
def create_messages(system_message, user_message):

    messages = [
            {"role": 'system', "content": system_message},
            {"role": 'user', "content": user_message}
    ]
    return messages


def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        fixed = re.sub(r'```(?:json)?|```', '', text).strip()
        return json.loads(fixed)
    
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  
)


def invoke_llm(messages, temperature=0.1):

    llm.temperature = temperature
    response = llm.invoke(messages)
    return safe_json_parse(response.content)