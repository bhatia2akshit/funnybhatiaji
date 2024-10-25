import pytest
from joke_bot import Bot
import pyjokes

@pytest.fixture
def bot():
    return Bot()

def test_tell_joke(bot):
    joke = bot.tell_joke()
    assert isinstance(joke, str), "Joke is not a string."

def test_rate_joke(bot):
    joke = pyjokes.get_joke()
    rating = bot.rate_joke(joke)
    assert isinstance(rating, (int, float)), "Rating is not a number."
    assert 0 <= rating <= 10, "Rating is not within the correct range."

