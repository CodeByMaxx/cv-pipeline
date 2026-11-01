import boto3

print("[*] Prüfe verfügbare Bedrock-Modelle für deinen Account...\n")
try:
    bedrock = boto3.client("bedrock", region_name="eu-central-1")
    response = bedrock.list_foundation_models(byProvider="Anthropic")

    models = response.get("modelSummaries", [])
    if not models:
        print(
            "[!] Keine Anthropic-Modelle gefunden. Vermutlich ist der Model Access in der AWS Konsole nicht aktiviert."
        )
    for m in models:
        print(f" - Model ID : {m['modelId']}")
        print(f"   Name     : {m.get('modelName')}")
        print(f"   Status   : {m.get('responseStreamingSupported', 'N/A')}\n")
except Exception as e:
    print(f"[!] Fehler beim Abrufen: {e}")
