# HF Pro Inference Playground

A Gradio app for chatting with models through Hugging Face's Inference
Providers router, spending your HF PRO monthly credits — with a model
picker, provider routing policy, and a running usage estimate.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Set your token

Create a Hugging Face access token with **"Make calls to Inference
Providers"** permission at https://huggingface.co/settings/tokens.

Recommended: put it in a local `.env` file (never commit this) so you
never have to paste it into the app:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder:

```
HF_TOKEN=hf_your_real_token
```

`app.py` loads this automatically via `python-dotenv`. The `.gitignore`
already excludes `.env`, so it will never end up in version control —
only `.env.example` (with a placeholder, no real secret) is meant to be
committed.

If you'd rather not use a `.env` file, you can instead export the
variable directly in your shell (`export HF_TOKEN=hf_xxx`) or paste the
token into the app's "HF Access Token" box each time you run it.

## Test locally

```bash
python app.py
```

This starts a local server (default `http://127.0.0.1:7860`). Open that
URL, pick a model, send a message, and confirm:

- the reply streams in
- the "Tokens this session" / estimated spend numbers update after each reply
- switching models and provider policy still works
- "Clear chat" resets the conversation and the usage panel back to zero

For auto-reload while you edit:

```bash
gradio app.py
```

Only once it behaves the way you want locally should you push it
somewhere (e.g. a Hugging Face Space) or commit it to git — the
`.gitignore` here already excludes your virtual environment and any
`.env` file so a token never ends up in version control.

## Note on the credit balance

Hugging Face doesn't expose an API for your exact real-time credit
balance. The panel in this app shows an *estimate* based on token counts
and rough public prices. For your real balance, check
https://huggingface.co/settings/billing
