"""Authentication helpers for the Signalium API client."""

from __future__ import annotations

import os

import requests

from trainer.api.exceptions import APIAuthenticationError


class BearerAuth(requests.auth.AuthBase):
    """Attach a Bearer token to every request."""

    def __init__(self, token: str) -> None:
        self._token = token

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers["Authorization"] = f"Bearer {self._token}"
        return r


class APIKeyAuth(requests.auth.AuthBase):
    """Attach an API key header to every request."""

    def __init__(self, api_key: str, header_name: str = "X-API-Key") -> None:
        self._api_key = api_key
        self._header_name = header_name

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers[self._header_name] = self._api_key
        return r


def resolve_auth(
    api_key: str | None = None,
    token: str | None = None,
) -> requests.auth.AuthBase:
    """Resolve authentication from explicit values or environment variables.

    Priority: explicit token > explicit api_key > env SIGNALIUM_TOKEN > env SIGNALIUM_API_KEY.

    Raises:
        APIAuthenticationError: When no credentials are found.
    """
    resolved_token = token or os.getenv("SIGNALIUM_TOKEN")
    if resolved_token:
        return BearerAuth(resolved_token)

    resolved_key = api_key or os.getenv("SIGNALIUM_API_KEY")
    if resolved_key:
        return APIKeyAuth(resolved_key)

    raise APIAuthenticationError(
        "No Signalium credentials found. "
        "Set SIGNALIUM_TOKEN or SIGNALIUM_API_KEY environment variables."
    )
