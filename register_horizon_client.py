
import json
import requests


REGISTRATION_URL = (
    "https://magic-beige-goat.fastmcp.app/oauth2/register"
)

CALLBACK_PORT = 53030

payload = {
    "client_name": "Rajendra Expense MCP Client",
    "application_type": "web",
    "redirect_uris": [
        f"http://localhost:{CALLBACK_PORT}/callback"
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

response.raise_for_status()

data = response.json()

# Save credentials locally.
with open("horizon_client.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("\nOAuth client registered successfully.")
print("Client ID:", data["client_id"])
print("Auth method:", data["token_endpoint_auth_method"])
print("Credentials saved to: horizon_client.json")

if "client_secret" in data:
    print("Client secret received: YES")
else:
    print("Client secret received: NO")

