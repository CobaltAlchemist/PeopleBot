import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import re
	
class PlayerModel(torch.nn.Module):
	def __init__(self, path, base, gpu = False, **kwargs):
		super().__init__()
		assert path is not None, "Must specify a model path"
		self.model = GPT2LMHeadModel.from_pretrained(None,
			config=base, state_dict=torch.load(path))
		self.tokenizer = GPT2Tokenizer.from_pretrained(base)
		device = 'cpu'
		if gpu and torch.cuda.is_available():
			device = 'cuda'
		self.model.eval()
		self.model.to(device)
		
	def respond(self, s):
		s = re.sub(r'<@!\d+>', '', s).strip()
		s = "<|conv|> " + s + " <|response|> "
		cur_ids = torch.tensor(self.tokenizer.encode(s)).unsqueeze(0).to(self.model.device)
		sample_output = self.model.generate(cur_ids, max_length=200, do_sample=True, top_k=5)
		output_text = self.tokenizer.decode(sample_output[0])
		output_text = output_text[len(s):]
		output_text = output_text.replace(self.tokenizer.eos_token, '')
		return output_text.strip()
		
if __name__ == "__main__":
	guillermo = PlayerModel(r'models\guillermo.pt')
		
	print(guillermo.respond("john#1029: Hello guillermo"))
	