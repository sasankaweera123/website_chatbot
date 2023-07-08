from chat import get_response


def test_get_response():
    # Test greetings
    assert get_response("Hi") == "Hello, how can I help you?" or "Hey :-)" or "Hi there, how can I help?" or\
           "Hello, what can I do for you?" or "Hi there, how can I help?"
    # Test goodbye
    assert get_response("Bye") == "See you later, thanks for visiting" or "Have a nice day" or\
           "Bye! Come back again soon." or "Thanks for visiting. See you later." or "Thanks, bye"
    # Test thanks
    assert get_response("Thanks") == "Happy to help!" or "Any time!" or "My pleasure" or\
           "You're most welcome!" or "Any time. Bye for now." or "You're welcome"
    # Test jokes
    assert get_response("Tell me a joke!") in [
        "Why did the hipster burn his mouth? He drank the coffee before it was cool.",
        "What did the buffalo say when his son left for college? Bison."
    ]
