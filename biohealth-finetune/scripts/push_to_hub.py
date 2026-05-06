# Share your fine-tuned adapter publicly (or privately)

import os
from peft import PeftModel
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

ADAPTER_DIR = './outputs/biohealth-ner-adapter'
HF_USERNAME = 'julien-g-man'
REPO_NAME   = f'{HF_USERNAME}/biohealth-ner-mistral7b-qlora'

api = HfApi(token=os.getenv('HF_TOKEN'))

# Create the repo if it doesn't exist
try:
    api.create_repo(repo_id=REPO_NAME, private=True)  # private=True for your sprint project
    print(f'Created repo: {REPO_NAME}')
except Exception as e:
    print(f'Repo may already exist: {e}')

# Upload the adapter files
api.upload_folder(
    folder_path=ADAPTER_DIR,
    repo_id=REPO_NAME,
    commit_message='Add QLoRA biomedical NER adapter',
)

print(f'Adapter pushed to: https://huggingface.co/{REPO_NAME}')
print('Anyone can now load your adapter with:')
print(f'  model = PeftModel.from_pretrained(base, "{REPO_NAME}")')
