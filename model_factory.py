import models

model_pool = {
    'baseline': models.BaseLine,
    'baseline2': models.BaseLine2
}

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    decoder_type = config_data['model'].get('decoder_type','LSTM')

    # You may add more parameters if you want
    return model_pool[model_type](vocab_size=len(vocab), embed_size=embedding_size, h_size=hidden_size, decoder_type=decoder_type)