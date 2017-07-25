import json

class Hps(object):
    def __init__(self,
                 lr=0.3,
                 hidden_dim=256,
                 embedding_dim=300,
                 keep_prob=0.8,
                 batch_size=32,
                 encoder_length=100,
                 decoder_length=15):
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length

    def load(self, path):
        with open(path, 'r') as f_json:
            hps_json = json.load(f_json)
        self.lr = hps_json['lr']
        self.hidden_dim = hps_json['hidden_dim']
        self.embedding_dim = hps_json['embedding_dim']
        self.keep_prob = hps_json['keep_prob']
        self.batch_size = hps_json['batch_size']
        self.encoder_length = hps_json['encoder_length']
        self.decoder_length = hps_json['decoder_length']

    def dump(self, path):
        hps_json = {
            'lr':self.lr, 
            'hidden_dim':self.hidden_dim,
            'embedding_dim':self.embedding_dim,
            'keep_prob':self.keep_prob,
            'batch_size':self.batch_size,
            'encoder_length':self.encoder_length,
            'decoder_length':self.decoder_length,
        }
        with open(path, 'w') as f_json:
            json.dump(hps_json, f_json, indent=4, separators=(',', ': '))
        

class Vocab(object):
    def __init__(self):
        self.word2idx = {'<PAD>':0, '<BOS>':1, '<EOS>':2}
    def size(self):
        return 5
