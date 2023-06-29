import torch.nn as nn
from ChatBot.config.configuration import ConfigurationManager
from ChatBot.components.train import *
import os
from ChatBot.models.EncoderRNN import EncoderRNN
from ChatBot.models.DecoderRNN import LuongAttnDecoderRNN
import torch
from torch import optim

config = ConfigurationManager()
config = config.get_train_config()

class TrainingPipeline:
    def __init__(self, voc, pairs):
        self.voc=voc
        self.pairs=pairs
    def initialize(self):
        loadFilename = os.path.join(config.save_dir, config.model_name, config.corpus_name,
                        '{}-{}_{}'.format(config.encoder_n_layers, config.decoder_n_layers, config.hidden_size),
                        '{}_checkpoint.tar'.format(config.checkpoint_iter))
        # Load model if a ``loadFilename`` is provided
        if config.loadFilename:
            
            # If loading on same machine the model was trained on
            checkpoint = torch.load(config.loadFilename)
            # If loading a model trained on GPU to CPU
            #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            self.voc.__dict__ = checkpoint['voc_dict']
        print('Building encoder and decoder ...')
        # Initialize word embeddings
        embedding = nn.Embedding(self.voc.num_words, config.hidden_size)
        if config.loadFilename:
            embedding.load_state_dict(embedding_sd)
        # Initialize encoder & decoder models
        encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
        decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size, self.voc.num_words, config.decoder_n_layers, config.dropout)
        print('Models built and ready to go!')
        if config.loadFilename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)
        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()
        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)
        if config.loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        # Run training iterations
        print("Starting Training!")
        trainIters(config.model_name, self.voc, self.pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   embedding, config.encoder_n_layers, config.decoder_n_layers, config.save_dir, config.n_iteration, config.batch_size,
                   config.print_every, config.save_every, config.clip, config.corpus_name, config.loadFilename)