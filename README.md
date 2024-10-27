# Welcome to the "AI Comedy Club" Challenge


This is part of an interview round at Konfuzio. I developed the bot: funnybhatiaji, located in the folder bots.
This bot calls huggingface inference client api to make a joke and evaluate a joke.

If a previous joke is given, then it extracts the topics from the joke (using Keybert) and a joke is generated.
To evaluate a joke, using prompt engineering, scores are calculated for 10 different parameters by LLM and then an average score is returned.

As compared to other bots which I could install (because I dont have GPU, I cant download the bots which are dependent upon GPU), I found out that funnybhatiaji performed better than the others.
