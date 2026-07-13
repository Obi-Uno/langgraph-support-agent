from dotenv import load_dotenv

# Load .env before any app module reads os.getenv at import time
# (app.agent resolves LLM_PROVIDER/AGENT_MODEL when it is imported).
load_dotenv()
