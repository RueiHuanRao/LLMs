# -*- coding: utf-8 -*-

import os
from getpass import getpass

from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain  # noqa
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE  # noqa
from langchain.chains.router import MultiPromptChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain

from config import APIKey

os.environ['HUGGINGFACEHUB_API_TOKEN'] = APIKey.HUGGINGFACEHUB_API_TOKEN.value or getpass("HUGGINGFACEHUB_API_TOKEN: ")  # noqa
os.environ["PINECONE_API_KEY"] = APIKey.PINECONE_API_KEY.value or getpass("PINECONE_API_KEY: ")  # noqa
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"


llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.3, "max_length": 128}
)


# --------- #
# LLM alone #
# --------- #
question = "What is the capital of France, output the answer only"
question = "evaluate the following review, 'I ordered Pizza Salami and it was awesome!', output the subject and sentiment only"  # noqa
llm.invoke(question)


# ------------ #
# LLM + Prompt #
# ------------ #
prompt_template = """
    Interprete the text and evaluate it.
    setiment: is the text in a positive, neutral or negative sentiment?
    subject: what subject is the text about? Show exactly one word

    Format the output as JSON with the following keys:
    sentiment
    subject

    text: {input}
"""
prompt = PromptTemplate.from_template(template=prompt_template)
chain = LLMChain(prompt=prompt, llm=llm)
chain.invoke(input="I ordered Pizza Salami and it was awesome!")  # noqa

prompt = PromptTemplate(
    input_variables=["input"],
    template=prompt_template
)
chain = LLMChain(prompt=prompt, llm=llm)
chain.invoke(input="I ordered Pizza Salami and it was awesome!")  # noqa


# --------------------- #
# LLM + Prompt + Parser #
# --------------------- #
sentiment_schema = ResponseSchema(name="sentiment", description="is the text in a positive, neutral or negative sentiment?")  # noqa
subject_schema = ResponseSchema(name="subject", description="what subject is the text about? Use exactly one word")  # noqa
price_schema = ResponseSchema(name="price", description="how expensive was the product? use None if no price provided. Show exactly the price only")  # noqa

response_schema = [sentiment_schema, subject_schema, price_schema]

parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schema)  # noqa
format_instructions = parser.get_format_instructions()
format_instructions

template = """
    Interprete the text and evaluate it.

    text: {input}

    {format_instructions}
"""
prompt = PromptTemplate.from_template(template=template)
messages = prompt.format_prompt(input="I ordered Pizza Salami for 9.99$ and it was awesome", format_instructions=format_instructions)  # noqa
response = llm.invoke(messages)
response

# If using ChatGPT models, the 'response' will be an AIMessage, then replace response with response.content  # noqa
response_dic = parser.parse(response)
response_dic


# --------------- #
# Multiple Chains #
# --------------- #
template_review = """
    evaluate the following review, '{review}', output the subject and sentiment only  # noqa
"""
prompt_review = PromptTemplate.from_template(
    template=template_review
)
chain_review = LLMChain(prompt=prompt_review, llm=llm)
chain_review.invoke("I ordered Pizza Salami and it was awesome")  # noqa

template_response = """
    You are a helpful bot and will recieve the ordered items and sentiment according to the review: {review}.  # noqa
    If customers were unsatisfied with the order, offer a real-world assistant for them; otherwise, just reply a grateful response sentence to them  # noqa
    output the answer only
"""
prompt_response = PromptTemplate.from_template(
    template=template_response
)
chain_response = LLMChain(llm=llm, prompt=prompt_response)
chain_response.invoke("I ordered Pizza Salami and it was awesome")  # noqa

overall_chain = SimpleSequentialChain(chains=[chain_review, chain_response], verbose=True)  # noqa
overall_chain.invoke("I ordered Pizza Salami and it was awesome")  # noqa

overall_chain.invoke("I ordered Pizza Salami and it was terrible")  # noqa


# ---------------- #
# Sequential Cahin #
# ---------------- #
prompt_review = PromptTemplate(template="You ordered {dish_name} and your experience was {experience}. Write a review", input_variables=["dish_name", "experience"])  # noqa
chain_review = LLMChain(llm=llm, prompt=prompt_review, output_key="review")

prompt_comment = PromptTemplate(template="Given the review {review}, write a follow-up comment:", input_variables=["review"])  # noqa
chain_comment = LLMChain(llm=llm, prompt=prompt_comment, output_key="comment")

prompt_summary = PromptTemplate(template="summarise the following review in a short sentence. review: {review}", input_variables=["review"])  # noqa
chain_summary = LLMChain(llm=llm, prompt=prompt_summary, output_key="summary")

overall_chain = SequentialChain(
    chains=[chain_review, chain_comment, chain_summary],
    input_variables=["dish_name", "experience"],
    output_variables=["review", "comment", "summary"]
)
overall_chain.invoke({"dish_name": "Pizza Salami", "experience": "it is awesome"})  # noqa


# --------------------------- #
# Multiple Prompts (Rounters) #
# --------------------------- #
template_positive = """you are a customer service AI and focusing on the positive aspects of the review. review {input}"""  # noqa
template_neutral = """you are a customer service AI and focusing on the neutral aspects of the review. review {input}"""  # noqa
template_negative = """you are a customer service AI and focusing on the negative aspects of the review. review {input}"""  # noqa

prompt_infos = [
    {
        "name": "positive",
        "desc": "Good for analysing positive sentiments",
        "template": template_positive
        },
    {
        "name": "neutral",
        "desc": "Good for analysing neutral sentiments",
        "template": template_neutral
        },
    {
        "name": "negative",
        "desc": "Good for analysing negative sentiments",
        "template": template_negative
        }
]

destination_chains = {}
for prompt_info in prompt_infos:
    name = prompt_info["name"]
    prompt = PromptTemplate(template=prompt_info["template"], input_variables=["input"])  # noqa
    chain = LLMChain(prompt=prompt, llm=llm)
    destination_chains[name] = chain
destination_chains

destinations = [f"{p['name']}:{p['desc']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)  # noqa
router_template

router_prompt = PromptTemplate.from_template(template=router_template, input_variabls=["input"], output_parser=RouterOutputParser())  # noqa
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
router_chain

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=destination_chains["neutral"],
    verbose=True
)
chain.invoke("I ordered a pizza and it was amazing!")  # noqa


# ------------------ #
# LangChain + Memory #
# ------------------ #
history = ChatMessageHistory()
history.add_user_message("hi")
history.add_ai_message("hello my human friend")
history.messages

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi")
memory.chat_memory.add_ai_message("hello my human friend")
memory.load_memory_variables({})

conversations = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)
conversations.invoke("Hi")

conversations.invoke("I need to know the capital of France")
