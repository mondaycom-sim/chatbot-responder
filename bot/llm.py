"""LLM response generation using Amazon Bedrock.

Handles chat completion requests through Bedrock's converse API
for the chatbot responder service.
"""
import json
import boto3

_bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

DEFAULT_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"


def generate_response(user_message: str, conversation_history: list = None) -> str:
    """Generate a chat response using Bedrock converse API."""
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": [{"text": user_message}]})

    response = _bedrock_client.converse(
        modelId=DEFAULT_MODEL,
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.8},
    )
    return response["output"]["message"]["content"][0]["text"]


def stream_response(user_message: str, model_id: str = DEFAULT_MODEL):
    """Stream a chat response from Bedrock for real-time display."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": user_message}],
    })
    response = _bedrock_client.invoke_model_with_response_stream(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            yield chunk["delta"].get("text", "")