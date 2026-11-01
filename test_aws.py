import os
import boto3
from botocore.exceptions import ClientError

# Zu testende Regionen und Modell-IDs
TEST_CONFIGS = [
    ("eu-central-1", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ("eu-central-1", "anthropic.claude-3-sonnet-20240229-v1:0"),
    ("us-east-1", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ("us-east-1", "anthropic.claude-3-haiku-20240307-v1:0"),
]

print("[*] Starte AWS Bedrock Modell-Verfügbarkeitstest...\n")

for region, model_id in TEST_CONFIGS:
    print(f"Teste Region: {region} | Modell: {model_id} ...", end=" ")
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Say 'OK'"}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0.0},
        )
        print("-> [ERFOLGREICH! ✓]")
        print(f"   Antwort: {response['output']['message']['content'][0]['text']}\n")
        break  # Stopp beim ersten funktionierenden Modell
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(f"-> [FEHLER: {error_code}]")
    except Exception as e:
        print(f"-> [FEHLER: {e}]")
