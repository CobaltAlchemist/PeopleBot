import pandas as pd
import glob
import regex
import string
from tqdm.auto import tqdm
from datetime import datetime
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
from transformers import AdamW, get_linear_schedule_with_warmup

EOS_TOKEN = '<|endoftext|>'

class DiscordDataset(Dataset):
	def __init__(self, corpus = 'corpus.txt'):
		super().__init__()
		if isinstance(corpus, str):
			with open('corpus.txt','r') as f:
				corpus = f.read()
			print("Reading corpus.txt")
			self.corpus = [x for x in corpus.split(EOS_TOKEN)]
		else:
			print("Using pre-prepared corpus")
			self.corpus = corpus
		
	def __len__(self):
		return len(self.corpus)

	def __getitem__(self, item):
		return self.corpus[item]

def write_convo(conv, authorconv):
	s='<|conv|> '
	s+='\n'.join(conv)
	astr = "\n".join(authorconv)
	s+=f'\n<|response|> {astr}'
	return s
	
def minutes(d1, d2):
	return (d2-d1).total_seconds() / 60

def get_corpus(file, user, convo_time, convo_length, bench = False, **kwargs):
	top = None
	doflush = False
	ret = []
	data = pd.read_csv(file,parse_dates=[2], infer_datetime_format = True)
	data = data[['AuthorID', 'Author','Content','Date']]
	data.dropna(subset=['Content'])
	conv = []
	authconv = []
	last = None
	lines = 0
	pbar = tqdm(data.iterrows(), total=len(data))
	for index, row in pbar:
		authid, author, content, date = row
		content = str(content)
		date = pd.to_datetime(date)
		
		if last is None:
			last = date
		if minutes(last, date) > convo_time:
			if len(authconv) > 0:
				lines += len(conv) + len(authconv)
				ret.append(write_convo(conv, authconv))
			conv = []
			authconv = []
			
		if authid == user:
			if len(conv) == 0 or 'http' in content:
				continue
			authconv.append(content)
			if top is not None:
				if top == 0:
					break
				else:
					top -= 1
		else:
			if len(authconv) > 0:
				lines += len(conv) + len(authconv)
				ret.append(write_convo(conv, authconv))
				conv = []
				authconv = []
				if bench and len(ret) > 20: break
			if 'http' not in content:
				conv.append(f"{author}: {content}")
				if len(conv) > convo_length:
					conv = conv[-convo_length:]
		last = date
		pbar.set_postfix({'Lines': lines})
	return ret
	
def choose_from_top(probs, n=5):
	ind = np.argpartition(probs, -n)[-n:]
	top_prob = probs[ind]
	top_prob = top_prob / np.sum(top_prob) # Normalize
	choice = np.random.choice(n, 1, p = top_prob)
	token_id = ind[choice][0]
	return int(token_id)
	
def respond(tokenizer, model, device, s):
	cur_ids = torch.tensor(tokenizer.encode(s)).unsqueeze(0).to(device)
	for i in range(100):
		logits = model(cur_ids, labels=cur_ids)[1]
		softmax_logits = torch.softmax(logits[0,-1], dim=0)
		if i < 3:
			n = 20
		else:
			n = 3
		next_token_id = choose_from_top(softmax_logits.cpu().numpy(), n=n)
		cur_ids = torch.cat([cur_ids, torch.tensor([[next_token_id]], device=device)], dim = 1)
		if next_token_id in tokenizer.encode(EOS_TOKEN):
			break
	output_list = list(cur_ids.squeeze().to('cpu').numpy())
	output_text = tokenizer.decode(output_list)
	return output_text
	
def make_models(corpus, name, epochs=10, seq_size=1024, learning_rate=3e-5, model='medium', bench=False, gpu=False, **kwargs):
	device = 'cpu'
	if torch.cuda.is_available() and gpu:
		device = 'cuda'
		
	if bench:
		epochs = 1
		seq_size=1

	model_size = model
	tokenizer = GPT2Tokenizer.from_pretrained(f'gpt2-{model_size}')
	if EOS_TOKEN != tokenizer.eos_token:
		print("Overriding eos with", EOS_TOKEN)
		tokenizer.eos_token = EOS_TOKEN
	model = GPT2LMHeadModel.from_pretrained(f'gpt2-{model_size}')
	model = model.to(device)
	dset = DiscordDataset(corpus)
	dl = DataLoader(dset, batch_size=1, shuffle=True)
	optimizer = AdamW(model.parameters(), lr=learning_rate)

	models_folder = "trained_models"
	if not os.path.exists(models_folder):
		os.mkdir(models_folder)
	
	for epoch in range(epochs):

		model.train()
		print(f"EPOCH {epoch} started" + '=' * 30)
		pbar = tqdm(enumerate(dl), total = len(dl))
		seq = []
		labels = []
		for idx, convo in pbar:
			conv, response = convo[0].split('<|response|>')
			conv += '<|response|>'
			convids = tokenizer(conv, max_length=seq_size+1, truncation=True)['input_ids']
			respids = tokenizer(response, max_length=seq_size+1, truncation=True)['input_ids']
			respids.extend(tokenizer.encode(tokenizer.eos_token))
			
			toklabels = [-100] * len(convids) + respids
			toks = convids + respids
			
			if len(toks) > seq_size:
				continue
				
			if len(seq) + len(toks) > seq_size:
				x = torch.tensor([seq], device=device)
				y = torch.tensor([labels], device=device)
				seq = []
				labels = []
				loss, logits, _  = model(x, labels=y)
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				model.zero_grad()
				pbar.set_postfix({'loss': loss.item(), 'seqlen': len(x[0])})
				if bench: break
			seq.extend(toks)
			labels.extend(toklabels)

		# Store the model after each epoch to compare the performance of them
		torch.save(
			{k: v.cpu() for k, v in model.state_dict().items()},
			os.path.join(models_folder, f"gpt2_{model_size}_{name}_{epoch}.pt")
		)
		model.eval()
		with torch.no_grad():
			prompts = [
				"<|conv|> AGiantPanda#0069: Can you fit your foot in your mouth? <|response|> ",
				"<|conv|> AngryPineapple#1029 These new AI are so screwed lol <|response|> ",
				"<|conv|> BeepBoop#2847: I just got the new pokemon gold <|response|> "
			]
			with open(os.path.join(models_folder, f"{name}.txt"), 'a+', encoding="utf-8") as f2:
				f2.write(f"EPOCH {epoch}\n")
				for p in prompts:
					cur_ids = torch.tensor(tokenizer.encode(p)).unsqueeze(0).to(device)
					sample_output = model.generate(cur_ids, max_length=256, do_sample=True, top_k=5)
					s = tokenizer.decode(sample_output[0])
					f2.write(s + '\n')
					print(s)
		
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Person Bot Runner')
	parser.add_argument('-u', '--user', type=int, help='User Key')
	parser.add_argument('-n', '--name', type=str, help='Common name for the user')
	parser.add_argument('-f', '--file', default='DregsLogs.csv', help='Chat log file')
	parser.add_argument('-e', '--epochs', default=10, type=int)
	parser.add_argument('--model', default='medium')
	parser.add_argument('--seq_size', default=512, type=int)
	parser.add_argument('--convo_time', default=15, type=int)
	parser.add_argument('--convo_length', default=10, type=int)
	parser.add_argument('--learning_rate', default=1e-5, type=float)
	parser.add_argument('--bench', action='store_true')
	parser.add_argument('--gpu', action='store_true')
	args = parser.parse_args()
	corpus = get_corpus(**vars(args))
	print(f"Gathered {len(corpus)} samples")
	make_models(corpus, **vars(args))