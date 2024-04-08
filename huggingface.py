# -*- encoding=utf-8 -*-

# from langchain.llms import huggingface_hub
from langchain_community.llms import HuggingFaceHub
from langchain.llms import huggingface_pipeline  # noqa
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from getpass import getpass
from config import APIKey

os.environ['HUGGINGFACEHUB_API_TOKEN'] = APIKey.HUGGINGFACEHUB_API_TOKEN.value or getpass("HUGGINGFACEHUB_API_TOKEN: ")  # noqa


def local_llm():
    """
    #############################
    # run Mistral model locally #
    #############################
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")  # noqa

    pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=128
    )
    llm = huggingface_pipeline(pipeline=pipeline)

    return llm


template = """Question: {question} \n Answer: Let's think step by step"""

prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.5, "max_length": 64})

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run("what is S4 in Language models?"))
