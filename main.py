import importlib.util
import os
import subprocess
from dotenv import load_dotenv

# The directory where the bots are located
bots_dir = './bots/'
load_dotenv()


def check_test_pass(directory):
    """Check whether the tests pass for the given directory."""
    # Check if a test_bot.py file exists in the directory
    if not os.path.exists(os.path.join(directory, 'test_bot.py')):
        return False

    # Run pytest on the directory
    result = subprocess.run(['pytest', directory], stdout=subprocess.PIPE)
    return result.returncode == 0


# Find all the bots
bot_directories = [d for d in os.listdir(bots_dir) if os.path.isdir(os.path.join(bots_dir, d))]

bots = []
for bot_dir in bot_directories:
    # Only add the bot if its tests pass

    if not check_test_pass(os.path.join(bots_dir, bot_dir)):
        print(f"Skipping our ai comedy guest '{bot_dir}' because it's tests do not pass.")
        continue

    # Dynamically load the bot's module
    spec = importlib.util.spec_from_file_location("bot", os.path.join(bots_dir, bot_dir, "joke_bot.py"))
    bot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot_module)

    # Create an instance of the bot and add it to the list
    if hasattr(bot_module, "Bot"):
        bot = bot_module.Bot()

        # this helps with one of the bot not having a .name attribute apparently
        if hasattr(bot, "name"):
            bots.append(bot)

        else:
            print(f"bot dir without .name att: {bot_dir}")
    else:
        print(f"ERROR: bot dir without .Bot att: {bot_module, bot_dir}")

# Scorecard for each bot
print(f"printing bots list: {bots}")
scorecard = {bot.name: [] for bot in bots}
# Let each bot tell a joke and rate the others' jokes
joke = ''  # this keeps track of previously made joke as funnybhatiaji joke extract keywords from the previous joke
for bot in bots:
    if joke and bot.name == 'funnybhatiaji':
        joke = bot.tell_joke(joke)
    else:
        joke = bot.tell_joke()
    print(f"\n'{bot.name}' tells a joke: {joke}")

    for other_bot in bots:
        if other_bot is not bot:
            rating = other_bot.rate_joke(joke)
            print(f"'{other_bot.name}' rates the joke a {rating} out of 10")
            # Add the rating to the scorecard
            scorecard[bot.name].append(rating)

# Display the scorecard
print("\nScorecard:")
for bot_name, ratings in scorecard.items():
    average_rating = sum(ratings) / len(ratings) if ratings else 0
    print(f"{bot_name}: {average_rating}")
