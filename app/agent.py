import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import BaseModel

load_dotenv()

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


class ClassificationOutput(BaseModel):
    category: str


classifier = LlmAgent(
    name="classifier",
    model="gemini-2.5-flash",
    instruction="Classify the user query into 'shipping_related' (rates, tracking, delivery, returns) or 'unrelated'. Only use these two exact string values.",
    output_schema=ClassificationOutput,
)


def save_query(node_input: types.Content):
    text = node_input.parts[0].text if node_input.parts else ""
    return Event(
        output=node_input, actions=EventActions(state_delta={"user_query": text})
    )


def route_query(ctx: Context, node_input: dict):
    category = node_input.get("category", "unrelated")
    user_query = ctx.state.get("user_query", "")
    content = types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    return Event(output=content, actions=EventActions(route=category))


shipping_faq = LlmAgent(
    name="shipping_faq",
    model="gemini-2.5-flash",
    instruction=(
        "You are a super enthusiastic and playful shipping FAQ agent! 🚢✨ "
        "Answer questions about shipping rates, tracking, delivery, and returns with lots of energy and emojis 📦🏃‍♂️💨! "
        "IMPORTANT: Always make sure to excitedly highlight our special deal: FREE shipping on all orders over $50! 🎉💸"
    ),
)


def decline_node(node_input: types.Content):
    msg = "I'm sorry, I can only assist with shipping-related questions like rates, tracking, delivery, and returns."
    return Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
        output=msg,
    )


root_agent = Workflow(
    name="customer_support",
    edges=[
        ("START", save_query),
        (save_query, classifier),
        (classifier, route_query),
        (route_query, {"shipping_related": shipping_faq, "unrelated": decline_node}),
    ],
    description="Customer support workflow",
)

app = App(
    root_agent=root_agent,
    name="app",
)
