
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.qa_engine import PandasAIClient, QAResult

class TestQARetry(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.client = PandasAIClient(api_key="fake_key")
        # Mock the OpenAI client
        self.client.client = MagicMock()

    def test_retry_logic_success_after_failure(self):
        """Test that it retries and eventually succeeds."""
        # Mock responses: 
        # 1. Bad code (raises syntax error or runtime error)
        # 2. Good code
        
        mock_response_1 = MagicMock()
        mock_response_1.choices[0].message.content = "```python\nprint(undefined_var)\n```"
        
        mock_response_2 = MagicMock()
        mock_response_2.choices[0].message.content = "```python\nst.write('Success')\n```"
        
        self.client.client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        result = self.client.ask(self.df, "Test question", explain=False)
        
        # Should have called API twice
        self.assertEqual(self.client.client.chat.completions.create.call_count, 2)
        # Should be successful eventually
        self.assertFalse(result.has_error)
        self.assertEqual(result.iterations_used, 2)
        self.assertIn("Success", result.st_components[0]['content'])

    def test_max_retries_reached(self):
        """Test that it stops after max retries."""
        # Mock responses: Always bad code
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "```python\nraise Exception('Fail')\n```"
        
        self.client.client.chat.completions.create.return_value = mock_response
        
        result = self.client.ask(self.df, "Test question", explain=False)
        
        # Should have called API 3 times (MAX_ITERATIONS)
        self.assertEqual(self.client.client.chat.completions.create.call_count, 3)
        # Should be error
        self.assertTrue(result.has_error)
        self.assertEqual(result.iterations_used, 3)

if __name__ == '__main__':
    unittest.main()
