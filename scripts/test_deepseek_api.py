#!/usr/bin/env python3
"""Smoke-test DeepSeek API through the OpenAI-compatible SDK."""

import os
import sys
import argparse
from pathlib import Path

import httpx
from dotenv import dotenv_values, load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test DeepSeek API through the OpenAI-compatible SDK."
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Do not use proxy variables from the project .env file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_values = dotenv_values(ENV_FILE)
    load_dotenv(ENV_FILE, override=True)

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"FAIL: {API_KEY_ENV} is not set in {ENV_FILE}", file=sys.stderr)
        return 1

    proxy_url = None
    if not args.no_proxy:
        proxy_url = (
            env_values.get("HTTPS_PROXY")
            or env_values.get("HTTP_PROXY")
            or env_values.get("ALL_PROXY")
            or None
        )
    http_client = httpx.Client(trust_env=False, proxy=proxy_url, timeout=30.0)
    client = OpenAI(api_key=api_key, base_url=BASE_URL, http_client=http_client)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise API smoke-test assistant.",
                },
                {
                    "role": "user",
                    "content": "Reply with a short confirmation that the DeepSeek API is usable.",
                },
            ],
            temperature=0,
            max_tokens=80,
        )
    except AuthenticationError as exc:
        print(f"FAIL: authentication failed for {BASE_URL}: {exc}", file=sys.stderr)
        return 1
    except BadRequestError as exc:
        print(f"FAIL: request was rejected by {BASE_URL}: {exc}", file=sys.stderr)
        return 1
    except RateLimitError as exc:
        print(f"FAIL: rate limited by {BASE_URL}: {exc}", file=sys.stderr)
        return 1
    except APIConnectionError as exc:
        print(f"FAIL: could not connect to {BASE_URL}: {exc}", file=sys.stderr)
        return 1
    except APIError as exc:
        print(f"FAIL: API error from {BASE_URL}: {exc}", file=sys.stderr)
        return 1

    message = response.choices[0].message.content or ""
    print("SUCCESS: DeepSeek API call completed")
    print(f"base_url: {BASE_URL}")
    print(f"model: {response.model or MODEL}")
    print(f"proxy: {'enabled via .env' if proxy_url else 'disabled'}")
    print(f"response: {message.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
