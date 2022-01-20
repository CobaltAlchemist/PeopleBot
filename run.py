import discord
import discord.utils as dutil
import os
from playermodel import PlayerModel
import argparse
import configparser

class PersonClient(discord.Client):
	def __init__(self, model):
		super().__init__(activity=discord.Game(name="@me a question!"))
		self.model = model
		
	async def on_ready(self):
		print('We have logged in as {0.user}'.format(client))
		
	async def on_message(self, message):
		try:
			if message.author == client.user:
				return
			if message.author.bot:
				return
			if len(message.content) == 0:
				return
			if not client.user in message.mentions:
				return
			
			s_guild = str(message.guild)
			s_author = str(message.author)
			s = message.content
			
			response = ""
			i = 0
			while len(response) == 0 and i < 5:
				async with message.channel.typing():
					response = model.respond(s_author + ": " + s)
				i+=1
			print(f"Message: {s}, Response: {response}")
			await message.channel.send(response)
			
		except Exception as e:
			await message.channel.send(str(e))

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Person Bot Runner')
	parser.add_argument('-m', '--model', help='API Key')
	parser.add_argument('-k', '--key', help='API Key', default=None)
	parser.add_argument('--gpu', action='store_true')
	args = parser.parse_args()
	
	config = configparser.ConfigParser()
	config.read('config.ini')
	if args.key is None:
		args.key = config[args.model]['key']
	
	model = PlayerModel(**{**vars(args), **config[args.model]})
	model.respond('test')
	client = PersonClient(model)
	client.run(args.key)