"""
Pydantic request/response schemas for the SHL Assessment Recommender API.

Defines the data contracts for the /chat endpoint including:
- Message: A single conversation turn (user or assistant)
- ChatRequest: The full conversation history sent by the client
- Recommendation: A single SHL assessment recommendation
- ChatResponse: The agent's reply with optional recommendations
"""

from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    """A single message in the conversation history.

    Attributes:
        role: Either "user" or "assistant".
        content: The text content of the message.
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """The request body for the /chat endpoint.

    The client sends the FULL conversation history with every request
    (the service is stateless — no server-side session storage).

    Attributes:
        messages: Ordered list of Messages representing the conversation so far.
    """
    messages: list[Message]


class Recommendation(BaseModel):
    """A single SHL assessment recommendation.

    Attributes:
        name: The exact assessment name from the SHL catalog.
        url: The exact product URL from the SHL catalog.
        test_type: Comma-separated key codes (e.g. "K", "P", "A,S").
    """
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """The response body returned by the /chat endpoint.

    Attributes:
        reply: The agent's conversational text response.
        recommendations: A list of 0-10 Recommendation objects.
                         Empty list [] when clarifying or refusing.
                         1-10 items when the agent commits to a shortlist.
        end_of_conversation: True only when the user confirms the shortlist
                             or says goodbye.
    """
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
