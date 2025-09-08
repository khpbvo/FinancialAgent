from unittest.mock import MagicMock, patch
import openai


def test_gps5_chat_completion():
    """Ensure we call the OpenAI client with the gpt-5 model."""
    with patch("openai.OpenAI") as MockClient:
        mock_client = MockClient.return_value
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Hello"
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response

        client = openai.OpenAI(api_key="test")
        result = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": "Say 'Hello'"}],
        )

        assert result.choices[0].message.content == "Hello"
        mock_client.chat.completions.create.assert_called_with(
            model="gpt-5",
            messages=[{"role": "user", "content": "Say 'Hello'"}],
        )
