import mxnet as mx
from mxnet import autograd, gluon, init, metric, nd
from mxnet.gluon import loss as gloss, nn, rnn
from mxnet.contrib import text



class SentimentNet(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,
                 bidirectional, num_outputs, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(
                len(vocab), embed_size, weight_initializer=init.Uniform(0.1))
            self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    input_size=embed_size)
            self.decoder = nn.Dense(num_outputs, flatten=False)

    def forward(self, inputs, begin_state=None):
        outputs = self.embedding(inputs)
        outputs = self.encoder(outputs)
        outputs = nd.concat(outputs[0], outputs[-1])
        outputs = self.decoder(outputs)
        return outputs
