
import requests
import streamlit as st


def ask_cloudflare(user_prompt: str, context: str = "") -> str:
    """Generate customer experience insights using Cloudflare Workers AI."""

    account_id = st.secrets["cloudflare"]["account_id"]
    api_token = st.secrets["cloudflare"]["api_token"]
    model = st.secrets["cloudflare"]["model"]

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model}"
    )

    system_prompt = (
        "You are CXInsight, a concise customer experience intelligence assistant. "
        "Help business owners make decisions quickly. "
        "Use only the supplied analytics as evidence. "
        "Never invent statistics or infer topic-level sentiment "
        "when topic-level sentiment data is unavailable. "
        "Return a maximum of 120 words. "
        "Use this format: "
        "1. Key finding; "
        "2. Evidence; "
        "3. Recommended action. "
        "Prioritize the three most important findings. "
        "Use short bullet points, not long paragraphs. "
        "Do not include lengthy introductions or conclusions."
    )

    full_prompt = user_prompt

    if context:
        full_prompt += (
            "\n\nCustomer analytics context:\n"
            + context
        )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        "max_completion_tokens": 250,

        # Disable reasoning mode so we receive a normal text answer.
        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90,
        )

        response.raise_for_status()
        data = response.json()

        if not data.get("success", False):
            raise RuntimeError(
                f"Cloudflare returned an error: {data.get('errors')}"
            )

        result = data.get("result", {})

        # Standard chat-completion response.
        choices = result.get("choices", [])

        if choices:
            message = choices[0].get("message", {})
            answer = message.get("content")

            if isinstance(answer, str) and answer.strip():
                return answer.strip()

        # Alternative Workers AI response format.
        answer = result.get("response")

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        raise RuntimeError(
            "The API request succeeded, but no text answer was returned. "
            f"Response fields: {list(result.keys())}"
        )

    except requests.RequestException as error:
        raise RuntimeError(
            f"Cloudflare API request failed: {error}"
        ) from error
