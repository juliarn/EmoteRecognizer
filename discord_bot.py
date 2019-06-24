import typing

import cv2
import discord
import requests
from discord.ext import commands

import config
from emote_recognizer import EmoteRecognizer

bot = commands.Bot(command_prefix="!")
emote_recognizer = EmoteRecognizer()


@bot.event
async def on_ready():
    avatar = bytes(requests.get(config.discord_bot_avatar).content)
    await bot.user.edit(username=config.discord_bot_name, avatar=avatar)

    game = discord.Game(config.discord_bot_game)
    await bot.change_presence(status=discord.Status.online, activity=game)

    print("Successfully launched the discord bot!")


@bot.command(name="emote")
async def emote_command(context: commands.Context, emote: typing.Union[discord.PartialEmoji, str]):
    if isinstance(emote, discord.PartialEmoji):
        emote.animated = False
        url = emote.url
    else:
        url = emote
    try:
        image_array = emote_recognizer.parse_image(url)
        emote_type = emote_recognizer.recognize(image_array)
        await context.send(embed=response_embed(True, f"This emote is more likely from type {emote_type.name}!", url))
    except cv2.error:
        await context.send(embed=response_embed(False, "Couldn't analyze this emote!"))


@emote_command.error
async def emote_command_error(context: commands.Context, _):
    await context.send(embed=response_embed(False, "Please provide a custom emote!"))


def response_embed(success, text, url=None):
    title = "Emote analyzed!" if success else "Error while analyzing."
    color = discord.Color.dark_green() if success else discord.Color.dark_red()

    embed = discord.Embed(title=title, description=text, color=color)
    embed.set_image(url=url) if url else None
    return embed


bot.run(config.discord_bot_token)
