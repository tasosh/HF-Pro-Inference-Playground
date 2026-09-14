"""
HF Pro Inference Playground
----------------------------
A Gradio app that lets you chat with models through Hugging Face's
Inference Providers router (the thing your HF PRO monthly credits pay for),
with a model/provider picker and a running usage estimate.

IMPORTANT ABOUT "CREDIT BALANCE":
Hugging Face does not currently expose a public API that returns your exact
remaining dollar balance. The only authoritative number lives on your
Billing dashboard: https://huggingface.co/settings/billing
This app therefore shows an ESTIMATED session spend (based on token counts
and known public per-token prices for a few models) and gives you a button
to jump straight to the real dashboard. Treat the in-app number as a rough
guide, not a source of truth.
"""

import os
import time
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file (if present) into os.environ

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Curated list of solid chat models available through the HF router.
# Format: display name -> (model_id, notes)
MODEL_CHOICES = {
    "Llama 3.3 70B Instruct (Meta)": "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen2.5 72B Instruct (Alibaba)": "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
    "Mistral Small 24B Instruct": "mistralai/Mistral-Small-24B-Instruct-2501",
    "Gemma 2 27B Instruct (Google)": "google/gemma-2-27b-it",
    "Phi-4 (Microsoft)": "microsoft/phi-4",
    "gpt-oss-120b (OpenAI, open weight)": "openai/gpt-oss-120b",
}

PROVIDER_POLICIES = {
    "Fastest (default)": "fastest",
    "Cheapest": "cheapest",
    "My preferred order (HF settings)": "preferred",
}

# Very rough public per-1M-token prices (USD) for a subset of models, just to
# give a ballpark session estimate. These vary by provider/quantization and
# WILL drift out of date — always treat as approximate.
ROUGH_PRICE_PER_1M_TOKENS = {
    "meta-llama/Llama-3.3-70B-Instruct": {"in": 0.35, "out": 0.40},
    "Qwen/Qwen2.5-72B-Instruct": {"in": 0.35, "out": 0.40},
    "deepseek-ai/DeepSeek-V3": {"in": 0.27, "out": 1.10},
    "mistralai/Mistral-Small-24B-Instruct-2501": {"in": 0.10, "out": 0.30},
    "google/gemma-2-27b-it": {"in": 0.20, "out": 0.20},
    "microsoft/phi-4": {"in": 0.10, "out": 0.10},
    "openai/gpt-oss-120b": {"in": 0.10, "out": 0.50},
}

BILLING_URL = "https://huggingface.co/settings/billing"
TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client(token: str) -> InferenceClient:
    """Build an InferenceClient routed through Hugging Face."""
    token = (token or os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise gr.Error(
            "No Hugging Face token found. Paste a token with 'Make calls to "
            f"Inference Providers' permission (create one at {TOKEN_SETTINGS_URL}), "
            "or set the HF_TOKEN environment variable."
        )
    return InferenceClient(token=token)


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = ROUGH_PRICE_PER_1M_TOKENS.get(model_id)
    if not prices:
        return 0.0
    return (prompt_tokens / 1_000_000) * prices["in"] + (completion_tokens / 1_000_000) * prices["out"]


def format_usage_panel(session_prompt_tokens, session_completion_tokens, session_cost, model_id):
    priced = model_id in ROUGH_PRICE_PER_1M_TOKENS
    cost_line = (
        f"**Estimated session spend:** ${session_cost:.4f}"
        if priced
        else "**Estimated session spend:** not available for this model (unpriced in this app)"
    )
    return (
        f"**Tokens this session:** {session_prompt_tokens:,} in / {session_completion_tokens:,} out\n\n"
        f"{cost_line}\n\n"
        f"*Estimate only — HF doesn't expose a live balance API. "
        f"Check your real credit balance on the [Billing dashboard]({BILLING_URL}).*"
    )


# ---------------------------------------------------------------------------
# Core chat function (streaming)
# ---------------------------------------------------------------------------

def respond(
    message,
    chat_history,
    token,
    model_choice,
    provider_policy,
    system_prompt,
    temperature,
    max_tokens,
    session_prompt_tokens,
    session_completion_tokens,
    session_cost,
):
    if not message or not message.strip():
        yield chat_history, format_usage_panel(
            session_prompt_tokens, session_completion_tokens, session_cost, ""
        ), session_prompt_tokens, session_completion_tokens, session_cost
        return

    model_id = MODEL_CHOICES[model_choice]
    policy = PROVIDER_POLICIES[provider_policy]
    routed_model = model_id if policy == "fastest" else f"{model_id}:{policy}"

    client = get_client(token)

    # chat_history is a list of {"role": ..., "content": ...} dicts (Chatbot type="messages")
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(chat_history)
    messages.append({"role": "user", "content": message})

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    try:
        stream = client.chat.completions.create(
            model=routed_model,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
            stream=True,
        )
    except Exception as e:
        chat_history[-1]["content"] = f"⚠️ Request failed: {e}"
        yield chat_history, format_usage_panel(
            session_prompt_tokens, session_completion_tokens, session_cost, model_id
        ), session_prompt_tokens, session_completion_tokens, session_cost
        return

    partial = ""
    last_usage = None
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            partial += delta
            chat_history[-1]["content"] = partial
            usage_panel = format_usage_panel(
                session_prompt_tokens, session_completion_tokens, session_cost, model_id
            )
            yield chat_history, usage_panel, session_prompt_tokens, session_completion_tokens, session_cost
        if getattr(chunk, "usage", None):
            last_usage = chunk.usage

    # Update running totals once the stream finishes.
    if last_usage:
        session_prompt_tokens += last_usage.prompt_tokens
        session_completion_tokens += last_usage.completion_tokens
    else:
        # Fallback: rough estimate (~4 chars/token) if the provider didn't return usage.
        session_prompt_tokens += max(1, len(message) // 4)
        session_completion_tokens += max(1, len(partial) // 4)

    session_cost += estimate_cost(
        model_id,
        last_usage.prompt_tokens if last_usage else max(1, len(message) // 4),
        last_usage.completion_tokens if last_usage else max(1, len(partial) // 4),
    )

    usage_panel = format_usage_panel(session_prompt_tokens, session_completion_tokens, session_cost, model_id)
    yield chat_history, usage_panel, session_prompt_tokens, session_completion_tokens, session_cost


def clear_session():
    return [], format_usage_panel(0, 0, 0.0, ""), 0, 0, 0.0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="HF Pro Inference Playground") as demo:
    gr.Markdown(
        "# 🤗 HF Pro Inference Playground\n"
        "Chat with models through **Hugging Face Inference Providers**, spending your "
        "PRO monthly credits. Pick a model, pick a provider strategy, and go."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=480, label="Chat")
            msg = gr.Textbox(
                placeholder="Type a message and press Enter…",
                label="Message",
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear chat")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            has_env_token = bool(os.environ.get("HF_TOKEN", "").strip())
            token_box = gr.Textbox(
                label="HF Access Token",
                placeholder=(
                    "Using HF_TOKEN from .env ✅ (leave blank)"
                    if has_env_token
                    else "hf_... (or set HF_TOKEN in a local .env file)"
                ),
                type="password",
                value="",
            )
            model_dd = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value=list(MODEL_CHOICES.keys())[0],
                label="Model",
            )
            provider_dd = gr.Dropdown(
                choices=list(PROVIDER_POLICIES.keys()),
                value="Fastest (default)",
                label="Provider routing policy",
            )
            system_box = gr.Textbox(
                label="System prompt (optional)",
                placeholder="You are a helpful assistant.",
                lines=2,
            )
            temperature_slider = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            max_tokens_slider = gr.Slider(64, 4096, value=1024, step=64, label="Max output tokens")

            gr.Markdown("### 💳 Credits & Usage")
            usage_panel = gr.Markdown(format_usage_panel(0, 0, 0.0, ""))
            gr.Button("Open real Billing Dashboard ↗", link=BILLING_URL)

    # Session-scoped state for running totals
    st_prompt_tokens = gr.State(0)
    st_completion_tokens = gr.State(0)
    st_cost = gr.State(0.0)

    respond_inputs = [
        msg, chatbot, token_box, model_dd, provider_dd, system_box,
        temperature_slider, max_tokens_slider,
        st_prompt_tokens, st_completion_tokens, st_cost,
    ]
    respond_outputs = [chatbot, usage_panel, st_prompt_tokens, st_completion_tokens, st_cost]

    msg.submit(respond, respond_inputs, respond_outputs).then(
        lambda: "", None, msg, queue=False
    )

    send_btn.click(respond, respond_inputs, respond_outputs).then(
        lambda: "", None, msg, queue=False
    )

    clear_btn.click(
        clear_session, None,
        [chatbot, usage_panel, st_prompt_tokens, st_completion_tokens, st_cost],
    )

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
