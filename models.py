import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
from torchvision.models import resnet50
# from torchvision.models import ResNet50_Weights

from device import device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def temperature_softmax(scores, T):
    # scores [batch, 1, vocab]
    return F.softmax(scores/T, dim=-1)


class Encoder(nn.Module):
    def __init__(self, out_size):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            # *list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children())[:-1],
            *list(resnet50(pretrained=True).children())[:-1],
        ).requires_grad_(False)

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
    decoders = {
        'LSTM': nn.LSTM,
        'RNN': nn.RNN,
    }

    def __init__(self, vocab_size, embed_size, h_size, decoder_type='LSTM'):
        super(Decoder, self).__init__()
        assert decoder_type in self.decoders
        self.decoder = self.decoders[decoder_type](
            input_size=embed_size, hidden_size=h_size,
            num_layers=2, batch_first=True,
        )
        self.proj = nn.Linear(h_size, vocab_size)

    def forward(self, f, hs=None):
        """
        input:  [batch, 1, embed_size]
        output {
                'logits': [batch, 1, vocab_size],
                'hidden_state': tuple(Tensor)
        }
        """
        o, h = self.decoder(f, hs)
        return self.proj(o), h


class BaseLine(nn.Module):
    def __init__(self, vocab_size, embed_size, h_size, decoder_type='LSTM'):
        super(BaseLine, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.h_size = h_size

        self.we = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(out_size=embed_size)
        self.decoder = Decoder(vocab_size, embed_size, h_size, decoder_type)

    def forward(self, input, caption):
        """
        input: [batch, 3, 256, 256]
        caption: [batch, max_seq_length]
        output: [batch, max_seq_length, vocab_size]
        """
        seq_len = caption.size(1)
        scores = torch.zeros((input.size(0), seq_len, self.vocab_size)).to(device)

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
    
    def generate(self, input, max_len=20, temperature=0.5):
        """
        input: [batch, 3, 256, 256]
        output: [batch, max_seq_length]
        """
        caption = torch.zeros((input.size(0), max_len),
                              dtype=torch.long).to(device)

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
    

class BaseLine2(BaseLine):
    def __init__(self, vocab_size, embed_size, h_size, decoder_type='LSTM'):
        super(BaseLine2, self).__init__(vocab_size, embed_size, h_size, decoder_type)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.h_size = h_size

        self.we = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(out_size=embed_size)
        self.decoder = Decoder(vocab_size, embed_size + 2048, h_size, decoder_type)

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
#         print("after encoder")
        o, h = self.decoder(encoder_embed)
        scores[:, 0, :] = o.squeeze()

      # Stage 1
        embed_caption = self.we(caption)  # [batch, max_seq_length, embed_size]
        for i in range(seq_len - 1):
            # no need to forward end/pad token
            o, h = self.decoder(embed_caption[:, i:i+1, :], h)
            scores[:, i+1, :] = o.squeeze()
        
        return scores

    def generate(self, input, max_len=20, temperature=0.5):
        """
        input: [batch, 3, 256, 256]
        output: [batch, max_seq_length]
        """
        caption = torch.zeros((input.size(0), max_len),dtype=torch.long).to(device)

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
    

class BaseLine2(BaseLine):
    def __init__(self, vocab_size, embed_size, h_size, decoder_type='LSTM'):
        super(BaseLine2, self).__init__(vocab_size, embed_size, h_size, decoder_type)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.h_size = h_size

        self.we = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(out_size=embed_size)
        self.decoder = Decoder(vocab_size, self.embed_size *2, h_size, decoder_type)

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
#         raise Exception(f'shape={encoder_embed.shape}')
#         print("after encoder")
        padding = torch.zeros(encoder_embed.shape).to(device)
        o, h = self.decoder(torch.cat((encoder_embed, padding), dim=2)) #[batch size, 600]#torch.cat((tensor_1, tensor_2), dim=1)
        scores[:, 0, :] = o.squeeze()
        

      # Stage 1
        # tf.concat([tensor_128_1_2048, tf.expand_dims(vector_128_2048, axis=1)], axis=1)
        embed_caption = self.we(caption)  # [batch, max_seq_length, embed_size]
        for i in range(seq_len - 1):
            # no need to forward end/pad token
            
            o, h = self.decoder(torch.cat((embed_caption[:, i:i+1, :],encoder_embed), axis=2), h)#self.decoder(embed_caption[:, i:i+1, :], h)

            scores[:, i+1, :] = o.squeeze()
        
        return scores

    def generate(self, input, max_len=20, temperature=0.5):
        """
        input: [batch, 3, 256, 256]
        output: [batch, max_seq_length]
        """
        caption = torch.zeros((input.size(0), max_len),dtype=torch.long).to(device)

        def sampling(output, i):
            p = temperature_softmax(output, temperature).squeeze()
            m = Categorical(p)
            caption[:, i] = m.sample()

        encoder_embed = self.encoder(input).view(input.size(0), 1, -1)
        padding = torch.zeros(encoder_embed.shape).to(device)
        o, h = self.decoder(torch.cat((encoder_embed, padding), dim=2))
        sampling(o, 0)

        for i in range(max_len - 1):
            embed_token = self.we(caption[:, i:i+1])
            o, h = self.decoder(torch.cat((embed_token, encoder_embed), axis=2), h)#(embed_token, h)
            sampling(o, i+1)
        return caption
