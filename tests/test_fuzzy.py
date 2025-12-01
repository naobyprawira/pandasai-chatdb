import pandas as pd
from rapidfuzz import fuzz

def _fuzzy_match(series: pd.Series, query: str, threshold: int = 85) -> pd.Series:
    """Return a boolean Series where True means the value fuzzy-matches the query."""
    
    query_lower = query.lower().strip()
    query_tokens = query_lower.split()
    
    def score(val):
        if pd.isna(val):
            return False
        val_str = str(val).lower().strip()
        
        # Strategy 1: Query is substring of value (exact partial match)
        if query_lower in val_str:
            return True
        
        # Strategy 2: All query tokens exist in value tokens
        val_tokens = val_str.split()
        if query_tokens and all(qt in val_tokens for qt in query_tokens):
            return True
        
        # Strategy 3: Fuzzy match the ENTIRE query against the START of value
        val_prefix = val_str[:len(query_lower) + 10]
        if fuzz.ratio(query_lower, val_prefix) >= threshold:
            return True
        
        # Strategy 4: Check if each query token fuzzy-matches any value token
        matched_tokens = 0
        for qt in query_tokens:
            for vt in val_tokens:
                if fuzz.ratio(qt, vt) >= threshold:
                    matched_tokens += 1
                    break
        if query_tokens and matched_tokens == len(query_tokens):
            return True
        
        return False
    
    return series.apply(score)

# Test Data
df = pd.DataFrame({
    'Supplier Name': ['DONG JIN TEXTILE', 'SUNG DONG', 'OTHER SUPPLIER', 'DONGG JIN (TYPO)', 'PT DONG JIN']
})

print("Testing 'DONG JIN'...")
mask = _fuzzy_match(df['Supplier Name'], 'DONG JIN', threshold=80)
print(df[mask])

print("\nTesting 'SUNG DONG'...")
mask = _fuzzy_match(df['Supplier Name'], 'SUNG DONG', threshold=80)
print(df[mask])
