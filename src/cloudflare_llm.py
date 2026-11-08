
import os
import tomllib
from pathlib import Path

import requests


def get_cloudflare_config():
    """Load Cloudflare credentials independently of the UI framework."""

    # Environment variables take priority, especially in deployment.
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    model = os.getenv(
        "CLOUDFLARE_MODEL",
        "@cf/zai-org/glm-4.7-flash"
    )

    # Local development fallback.
    if not account_id or not api_token:
        secrets_path = (
            Path(__file__).resolve().parent.parent
            / ".streamlit"
            / "secrets.toml"
        )

        if secrets_path.exists():
            with secrets_path.open("rb") as file:
                secrets = tomllib.load(file)

            cloudflare = secrets.get("cloudflare", {})

            account_id = account_id or cloudflare.get("account_id")
            api_token = api_token or cloudflare.get("api_token")
            model = cloudflare.get("model", model)

    if not account_id or not api_token:
        raise RuntimeError(
            "Cloudflare credentials are missing. "
            "Configure environment variables or local secrets."
        )

    return account_id, api_token, model


def ask_cloudflare(user_prompt: str, context: str = "") -> str:
    """Generate customer intelligence using Cloudflare Workers AI."""

    account_id, api_token, model = get_cloudflare_config()

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model}"
    )

    system_prompt = (
        "You are SXOUND, a customer intelligence assistant. "
        "Help business owners make decisions using customer analytics. "
        "Use only the supplied evidence. Never invent statistics. "
        "Be concise, clear and actionable. "
        "Limit answers to 120 words unless more detail is requested. "
        "Do not infer topic-level sentiment beyond the supplied data."
    )

    full_prompt = user_prompt.strip()

    if context:
        full_prompt += (
            "\n\nCustomer analytics context:\n"
            + context
        )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        "max_completion_tokens": 250,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90
        )

        response.raise_for_status()
        data = response.json()

        if not data.get("success", False):
            raise RuntimeError(
                f"Cloudflare error: {data.get('errors')}"
            )

        result = data.get("result", {})

        choices = result.get("choices", [])

        if choices:
            answer = choices[0].get("message", {}).get("content")

            if isinstance(answer, str) and answer.strip():
                return answer.strip()

        answer = result.get("response")

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        raise RuntimeError(
            f"No text answer returned. Response fields: "
            f"{list(result.keys())}"
        )

    except requests.RequestException as error:
        raise RuntimeError(
            f"Cloudflare API request failed: {error}"
        ) from error
