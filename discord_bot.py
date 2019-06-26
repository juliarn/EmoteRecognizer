import typing

import cv2
import discord
import requests
from discord.ext import commands

import config
from emote_recognizer import EmoteRecognizer

bot = commands.Bot(command_prefix=config.discord_bot_command_prefix)
emote_recognizer = EmoteRecognizer()


@bot.event
async def on_ready():
    """
    Changes the presence of the bot as defined in the config
    """

    avatar = bytes(requests.get(config.discord_bot_avatar).content)
    await bot.user.edit(username=config.discord_bot_name, avatar=avatar)

    game = discord.Game(config.discord_bot_game)
    await bot.change_presence(status=discord.Status.online, activity=game)

    print("Successfully launched the discord bot!")


@bot.command(name="emote")
async def emote_command(context: commands.Context, emote: typing.Union[discord.PartialEmoji, str]):
    """
    Handles the emote command

    :param context: the command context
    :param emote: the emote object or an url to the image of the emote
    """

    if isinstance(emote, discord.PartialEmoji):  # the user has provided a custom emote
        emote.animated = False  # make sure the image of the emote is not animated
        url = emote.url
    else:  # the user has provided an url-string
        url = emote
    try:
        image_array = emote_recognizer.parse_image(url)
        emote_type = emote_recognizer.predict(image_array)

        await context.send(embed=response_embed(True, f"This emote is more likely from type {emote_type.name}!", url))
    except cv2.error:
        await context.send(embed=response_embed(False, "Couldn't analyze this emote!"))


@emote_command.error
async def emote_command_error(context: commands.Context, _):
    """
    Handles the emote command executed with wrong arguments

    :param context: the command context
    :param _: the error object
    """

    await context.send(embed=response_embed(False, "Please provide a custom emote!"))


def response_embed(success, text, url=None):
    """
    Creates a response embed which will be send to the user

    :param success: if the emote could be analyzed
    :param text: the text of the embed
    :param url: the url for an image on the embed
    :return: the new embed
    """

    title = "Emote analyzed!" if success else "Error while analyzing."
    color = discord.Color.dark_green() if success else discord.Color.dark_red()

    embed = discord.Embed(title=title, description=text, color=color)
    embed.set_image(url=url) if url else None
    return embed


bot.run(config.discord_bot_token)
