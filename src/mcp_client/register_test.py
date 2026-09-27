
import requests
import json

REGISTRATION_URL = (
    "https://magic-beige-goat.fastmcp.app/oauth2/register"
)

payload = {
    "client_name": "Rajendra FastMCP Confidential Test",
    "redirect_uris": [
        "http://localhost:53030/callback"
    ],
    "grant_types": [
        "authorization_code",
        "refresh_token"
    ],
    "response_types": [
        "code"
    ],
    "token_endpoint_auth_method": "client_secret_post"
}

response = requests.post(
    REGISTRATION_URL,
    json=payload,
    timeout=30,
)

print("STATUS:", response.status_code)

print("\nBODY:")
try:
    data = response.json()

    # Don't accidentally print a secret.
    if "client_secret" in data:
        data["client_secret"] = "[REDACTED]"

    print(json.dumps(data, indent=2))

except Exception:
    print(response.text)

