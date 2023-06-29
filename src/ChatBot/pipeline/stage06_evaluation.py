import torch
import pathlib
from ChatBot.components.evaluation import *
from ChatBot.models.DecoderRNN import LuongAttnDecoderRNN
from ChatBot.models.EncoderRNN import EncoderRNN
from ChatBot.models.LuongAttention import Attn

class Evaluate:
    def __init__(self):
        pass
    @staticmethod
    def main(voc):
        hidden_size = 500
        encoder_n_layers = 2
        decoder_n_layers = 2
        dropout = 0.1
        batch_size = 64
        attn_model='dot'
        checkpoint = torch.load(pathlib.Path('artifacts/save/cb_model/movie-corpus/2-2_500/4000_checkpoint.tar'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
        # Initialize word embeddings
        embedding = nn.Embedding(voc.num_words, hidden_size)
        embedding.load_state_dict(embedding_sd)
        # Initialize encoder & decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        # Set dropout layers to ``eval`` mode
        encoder.eval()
        decoder.eval()
        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)
        return encoder, decoder, searcher
        # Begin chatting (uncomment and run the following line to begin)
        # evaluateInput(encoder, decoder, searcher, voc)