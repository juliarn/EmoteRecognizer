# EmoteRecognizer

The EmoteRecognizer is a discord bot deciding if an emote is more likely a kappa or a peepo emote.
This repo also contains a trained model and all the training data used.

# How to use

First, clone the repo:
```bash
$ git clone https://github.com/realPanamo/EmoteRecognizer
````
Edit the config.py file the way you want to and change 
the ``discord_bot_token`` to the token of your bot.

The following modules need to be installed via pip:

*  keras
*  opencv-python
*  numpy
*  requests
*  discord.py

Now you can start the discord bot:
```bash
$ python discord_bot.py
```
You should use a recent python version.
You can now use the !emote command on your discord to validate an emote.
