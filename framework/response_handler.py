# framework/response_handler.py

def extract_generated_text(response):
    """Extract text from pipeline response"""
    return response[0]["generated_text"]

def validate_response(generated_text, expected_substring=None):
    """Validate response from pipeline"""
    assert len(generated_text) > 0

    if expected_substring:
        assert expected_substring in generated_text, \
            f"Expected '{expected_substring}' in response"

    print(f"\nModel response: {generated_text}")