import boto3
from botocore.exceptions import ClientError

# Die Liste der Kandidaten basierend auf deiner AWS-Ausgabe
candidate_models = [
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-sonnet-5-5",
    "anthropic.claude-sonnet-5-5",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
]

print("[*] Teste AWS Bedrock Modelle im Detail...\n")

client = boto3.client("bedrock-runtime", region_name="eu-central-1")

for model_id in candidate_models:
    print(f"Teste ID: {model_id} ...", end=" ")
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Say 'OK'"}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0.0},
        )
        print("-> [LÄUFT! ✓]")
        print(f"   Antwort: {response['output']['message']['content'][0]['text']}\n")
        break  # Stopp beim ersten funktionierenden
    except ClientError as e:
        err_code = e.response["Error"]["Code"]
        print(f"-> [{err_code}]")
    except Exception as e:
        print(f"-> [FEHLER]")
