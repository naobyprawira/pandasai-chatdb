import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.qa_engine import PandasAIClient
from app.settings import AppSettings

# Load env
load_dotenv()

def test_openai_query():
    print("Loading data...")
    # Create dummy data since real file is corrupted
    print("Creating dummy data...")
    data = {
        'Supplier Name': ['EVER TECH CO LTD', 'OTHER SUPPLIER', 'EVER TECH INDONESIA', 'A B C'],
        'Saving Amount': [1000000, 500000, 2000000, 100000],
        'Saving %': [0.10, 0.05, 0.15, 0.01],
        'Year': [2025, 2025, 2025, 2024]
    }
    df = pd.DataFrame(data)
    print(f"Dummy data created: {len(df)} rows")

    print("Initializing PandasAIClient...")
    print(f"DEBUG: settings.default_llm_model = {AppSettings.default_llm_model}")
    try:
        client = PandasAIClient()
        print(f"DEBUG: client.model_name = {client.model_name}")
        
        # Test direct connection
        print("\n2. Testing direct OpenAI connection...")
        response = client.client.responses.create(
            model=client.model_name,
            instructions="You are a helpful assistant.",
            input="Say hello",

        )
        print(f"✅ Connection successful! Response: {response.output_text}")
        
    except Exception as e:
        print(f"Error initializing client or connecting: {e}")
        return

    query = "berapa total saving cost dan %age nya dari supplier EVER*TECH* selama tahun 2025?"
    print(f"\nRunning query: {query}")

    try:
        result = client.ask(df, query)
        
        print("\n--- RESULT ---")
        print(f"Response: {result.response}")
        print(f"Code:\n{result.code}")
        print(f"Explanation: {result.explanation}")
        
        if result.has_error:
            print(f"\n❌ FAILED with error: {result.failed_code}")
        else:
            print("\n✅ SUCCESS")
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_openai_query()
