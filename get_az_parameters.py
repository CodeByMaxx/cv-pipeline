python3 -c "
import os
print('--- Azure & LLM .env Setup ---')
azure_key = input('Azure OpenAI API Key (Enter für überspringen): ').strip()
azure_endpoint = input('Azure Endpoint (z.B. https://...): ').strip()
azure_dep = input('Azure Deployment Name [Standard: gpt-4o-mini]: ').strip() or 'gpt-4o-mini'
gemini_key = input('Gemini API Key (Enter für überspringen): ').strip()

env_lines = [
    f'AZURE_OPENAI_API_KEY={azure_key}',
    f'AZURE_OPENAI_ENDPOINT={azure_endpoint}',
    f'AZURE_OPENAI_DEPLOYMENT_NAME={azure_dep}',
    'AZURE_OPENAI_API_VERSION=2024-08-01-preview',
    'AWS_REGION=eu-central-1',
    'AWS_BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
    f'GEMINI_API_KEY={gemini_key}'
]

with open('.env', 'w', encoding='utf-8') as f:
    f.write('\n'.join(env_lines) + '\n')
print('[✓] .env Datei wurde erfolgreich im aktuellen Verzeichnis erstellt!')
"
