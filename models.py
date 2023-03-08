import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.models import resnet50
# from torchvision.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def temperature_softmax(scores, T):
    # scores [batch, 1, vocab]
    return F.softmax(scores/T, dim=-1)

class Encoder(nn.Module):
	def __init__(self, out_size):
		super(Encoder, self).__init__()

		self.encoder = nn.Sequential(
# 			*list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-1],
            *list(resnet50(pretrained=True).children())[:-1],
		)
		for param in self.encoder.parameters():
			param.requires_grad = False

		self.bridge = nn.Linear(2048, out_size)
		self.bn = nn.BatchNorm1d(out_size)

	def forward(self, input):
		"""
		input  [batch, 3, 256, 256]
		output [batch, out_dim]
		"""
		f = self.encoder(input).view(input.size(0), -1)
		return self.bn(self.bridge(f))


class Decoder(nn.Module):
	def __init__(self, vocab_size, embed_size, h_size):
		super(Decoder, self).__init__()
		self.l = nn.LSTM(input_size=embed_size,
						 hidden_size=h_size, num_layers=2, batch_first=True)
		self.proj = nn.Linear(h_size, vocab_size)

	def forward(self, f, hs=None):
		"""
		input:  [batch, 1, embed_size]
		output {
						'logits': [batch, 1, vocab_size],
						'hidden_state': tuple(Tensor)
				}
		"""
		o, h = self.l(f, hs)
		return self.proj(o), h


class BaseLine(nn.Module):
	def __init__(self, vocab_size, embed_size, h_size):
		super(BaseLine, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.h_size = h_size

		self.we = nn.Embedding(vocab_size, embed_size)
		self.encoder = Encoder(out_size=embed_size)
		self.decoder = Decoder(vocab_size, embed_size, h_size)

	def forward(self, input, caption):
		"""
		input: [batch, 3, 256, 256]
		caption: [batch, max_seq_length]
		output: [batch, max_seq_length, vocab_size]
		"""
		seq_len = caption.size(1)
		scores = torch.zeros(
			(input.size(0), seq_len, self.vocab_size)).to(device)

		# Stage 0
		encoder_embed = self.encoder(input).view(input.size(0), 1, -1)
		o, h = self.decoder(encoder_embed)
		scores[:, 0, :] = o.squeeze()

		# Stage 1
		embed_caption = self.we(caption)  # [batch, max_seq_length, embed_size]
		for i in range(seq_len - 1):
			# no need to forward end/pad token
			o, h = self.decoder(embed_caption[:, i:i+1, :], h)
			scores[:, i+1, :] = o.squeeze()

		return scores

	def generate(self, input, max_len = 20, temperature=0.5):
		"""
		input: [batch, 3, 256, 256]
		output: [batch, max_seq_length]
  		"""
		caption = torch.zeros((input.size(0), max_len), dtype=torch.long).to(device)
		def sampling(output, i):
			p = temperature_softmax(output, temperature).squeeze()
			m = Categorical(p)
			caption[:, i] = m.sample()
  
		encoder_embed = self.encoder(input).view(input.size(0), 1, -1)
		o, h = self.decoder(encoder_embed)
		sampling(o, 0)
  
		for i in range(max_len - 1):
			embed_token = self.we(caption[:, i:i+1])
			o, h = self.decoder(embed_token, h)
			sampling(o, i+1)
		return caption