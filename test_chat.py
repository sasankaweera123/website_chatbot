"""
This module contains unit tests for the chatbot's response generation.

The tests verify the expected responses for different input messages.

Usage:
    - Run the tests using a test runner or the pytest command.
"""

from chat import get_response


def test_get_response():
    """
    Test case for the get_response function.

    The test cases check the expected responses for different input messages.

    Raises:
        AssertionError: If any of the test assertions fail.
    """
    # Test greetings
    assert get_response("Hi") in [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?",
        "I'm glad you're talking to me.",
        "Hello, how can I help you?",
        "Hi, how can I help you?"
    ]

    # Test goodbye
    assert get_response("Bye") in [
        "See you later, thanks for visiting",
        "Have a nice day",
        "Bye! Come back again soon.",
        "Have a nice day. See you next time.",
        "Bye! Have a nice day.",
        "Bye! Have a nice day. See you next time."
    ]

    # Test thanks
    assert get_response("Thanks") in [
        "Happy to help!",
        "Any time!",
        "My pleasure",
        "You're most welcome!",
        "Any time. Bye for now.",
        "You're welcome"
    ]

    # Test jokes
    assert get_response("Tell me a joke!") in [
        "Why did the hipster burn his mouth? He drank the coffee before it was cool.",
        "What did the buffalo say when his son left for college? Bison."
    ]
