import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

# check cuda availability
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class Encoder(nn.Module):
    """
    The class that implements the Encoder.
    """

    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_len, embedding_size).to(device)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers).to(device)
        optim_parameters = {'lr': 1e-5, 'weight_decay': 1e-5}
        self.optim = Adam(self.parameters(), **optim_parameters)
        # init weights
        for _, p in self.state_dict().items():
            nn.init.normal_(p)

    def forward(self, input_seq):
        """
        Encodes the input sequence and outputs the hidden state and the cell state.
        :param input_seq: the sequence of words indices to be encoded
        :return: hidden state and cell state
        """
        seq_embedded = self.embedding(input_seq)  # of shape (seq_len, batch_size, embedding_size)
        output, (h_n, c_n) = self.lstm(seq_embedded)
        return h_n, c_n


class EncoderAttention(nn.Module):
    """
    The class that implements the Encoder which uses the attention mechanism
    """
    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1, max_length=10):
        super(EncoderAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(voc_len, embedding_size).to(device)
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True).to(device)
        self.max_length = max_length
        optim_parameters = {'lr': 1e-5, 'weight_decay': 1e-5}
        self.optim = Adam(self.parameters(), **optim_parameters)

        # init weights
        for _, p in self.state_dict().items():
            nn.init.normal_(p)

    def forward(self, input_seq):
        """
        Encodes the input sequence and outputs the hidden state and the cell state.
        :param input_seq: the sequence of words indices to be encoded
        :return: the output features from the last layer, hidden state and cell state
        """
        seq_embedded = self.dropout(self.embedding(input_seq))
        # forward pass through the encoder the sentence
        encoder_outputs, (h_n, c_n) = self.lstm(seq_embedded)
        # because of the bidirection both h_n and c_n have 2 tensors (forward, backward), but
        # the decoder is not bidirection, thus we merge the values from forward and backward direction.
        h_n = h_n[0:1, :, :] + h_n[1:2, :, :]
        c_n = c_n[0:1, :, :] + c_n[1:2, :, :]
        # similarly, we sum bidirectional values for the outputs
        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        return encoder_outputs, h_n, c_n


class Decoder(nn.Module):
    """
    Implements the Decoder of the Seq2Seq model
    """
    def __init__(self, embedding_size, hidden_size, voc_len, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(voc_len, embedding_size).to(device)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=False).to(device)
        self.relu = nn.ReLU().to(device)
        self.fc_1 = nn.Linear(hidden_size, voc_len).to(device)
        optim_parameters = {'lr': 1e-5, 'weight_decay': 1e-5}
        self.optim = Adam(self.parameters(), **optim_parameters)
        # init weights
        for _, p in self.state_dict().items():
            nn.init.normal_(p)

    def forward(self, x, h, c):
        """
        The forward method for the decoder.
        :param x: the input word embedded to be processed by the decoder. Initially is '<S>'
        :param h: the previous hidden state. Initially is the hidden state from the encoder's output
        :param c: the previous cell state. Initially is the cell state from the encoder's output
        :return output, (h_n, c_n):
        """
        # at this stage x has the dimension of (64). Hovewer it must be (1,64)
        x = x.unsqueeze(0)
        embedded_word = self.embedding(x)
        output, (h, c) = self.lstm(embedded_word, (h, c))
        output = self.relu(output)
        pred = self.fc_1(output).squeeze(0)
        return pred, h, c


class Attention(nn.Module):
    """
    Implements the attention mechanism.
    """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # hidden_size*2 is given by the fact that we concatenate
        # the prev_hidden (of size hidden_size) with the encoder states
        self.attn = nn.Linear(self.hidden_size * 2, 1).to(device)
        self.tanh = nn.Tanh().to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        # init weights
        for _, p in self.state_dict().items():
            nn.init.normal_(p)

    def forward(self, prev_hidden, encoder_outputs):
        """
        This function computes the attention weights which will be used by the model
        to focus more on a specific word in the sequence.
        :param prev_hidden: the previous hiddes state of the decoder (s_t-1)
        :param encoder_outputs: the outputs from the encoder (H)
        :return attention: the vector of shape [seq_len, batch] that contains the probabilities which reflects the
        level of 'relevance' for each word.
        """
        # encoder_outputs -> [seq_len, batch, hidden_size*2]
        # prev_hidden -> [seq_len, batch, hidden_size]
        # concatenate the previous hidden state and the encoder's output
        input_concat = torch.cat((prev_hidden, encoder_outputs), dim=2)
        # compute the energy values through the 'small' neural network attention.
        energy = self.tanh(self.attn(input_concat))
        # compute attention weights
        attention = self.softmax(energy.squeeze(2).t())
        return attention


class DecoderAttention(nn.Module):
    """
    Implements the Decoder with attention mechanism.
    """
    def __init__(self, embedding_size, hidden_size, voc_len, attention, n_layers=1):
        super(DecoderAttention, self).__init__()
        self.embedding = nn.Embedding(voc_len, embedding_size).to(device)
        self.hidden_size = hidden_size
        self.attention = attention
        self.lstm = nn.LSTM(self.hidden_size + embedding_size, hidden_size, num_layers=n_layers).to(device)
        self.fc_model = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size, voc_len)).to(device)
        optim_parameters = {'lr': 1e-5, 'weight_decay': 1e-5}
        self.optim = Adam(self.parameters(), **optim_parameters)

        # init weights
        for _, p in self.state_dict().items():
            nn.init.normal_(p)

    def forward(self, word, prev_hidden, prev_cell, encoder_outputs):
        """
        Decodes the sequence applying the attention mechanism.
        :param word: the word per batch taken in input by the Decoder. They can be either generated words or
        ground truth words depending on the teaching force probability.
        :param prev_hidden: the previous hidden state (s_t-1) of the Decoder
        :param prev_cell: the previous cell state of the Decoder
        :param encoder_outputs: the Encoder's output.
        :return predictions, h, c: the predictions for the next word; the new hidden and cell state.
        """
        # word -> [1, batch_size]

        # add first dimension to perform the embedding
        word = word.unsqueeze(0)
        # embed each word of the batch
        embedded = self.embedding(word)
        # init decoder's hidden state as the last encoder hidden state.
        prev_hidden_repeated = prev_hidden.repeat(encoder_outputs.shape[0], 1, 1)
        # compute the attention values that will specify what part of the input sentence is more relevant
        attention = self.attention(prev_hidden_repeated, encoder_outputs)
        # in order to combine the attention weights with the encoder outputs we want to multiply
        # them element wise. To do that we have to adjust the dimensions of those data structures.
        # the multiplication is achieved by the 'torch.bmm' operator. This will output the new context vector.
        attention = attention.unsqueeze(1)  # -> [batch, 1, seq_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # -> [batch, seq_len, hidden*2]
        context_vector = torch.bmm(attention, encoder_outputs)  # -> [batch, 1, hidden*2]
        context_vector = context_vector.permute(1, 0, 2)  # -> [1, batch, hidden*2]
        # Finally, concatenate the context vector and the embedded word to build the
        # new input to feed forward pass to the lstm network.
        new_input = torch.cat((context_vector, embedded), dim=2)
        outputs, (prev_hidden, prev_cell) = self.lstm(new_input, (prev_hidden, prev_cell))
        # to get the predictions feed forward pass the outputs from the decoder to a final fully connected layer.
        predictions = self.fc_model(outputs)
        predictions = predictions.squeeze(0)
        # we don't apply Softmax here because it will be applyied by the CrossEntropyLoss class.
        return predictions, prev_cell, prev_cell


class ChatbotModel(nn.Module):

    def __init__(self, encoder, decoder, vocab_size, with_attention, tf_ratio):
        super(ChatbotModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.attention = with_attention
        self.tf_ratio = tf_ratio

    def forward(self, X, y):
        """
        This function implement the forward method for both the regular Decoder and the
        one with the Attention mechanism implemented. This aspect is handled by a simple else-if control.
        In order to achieve a better convergence, the teaching force technique is used with probability of 0.5
        :param X: the batch of input sentences.
        :param y: the batch of target senteces.
        :return outputs: the outputs of the Decoder for each batch.
        """
        seq_len = y.shape[0]
        batch_size = X.shape[1]
        # this will store all the outputs for the batches, shape [seq_len, batch_size, len_vocab]
        outputs = torch.zeros(seq_len, batch_size, self.vocab_size, dtype=torch.float).to(device)
        # compute the hidden and cell state from the encoder
        if self.attention:
            # encode input sequence
            encoder_outputs, h_n, c_n = self.encoder(X)
            # set the first input to the SOS
            word_t = y[0]
            for t in range(1, seq_len):
                # decode sequence with attention
                output, h_n, c_n = self.decoder(word_t, h_n, c_n, encoder_outputs)
                # store logits
                outputs[t] = output
                # pick the best word from the vocabulary for each batch
                prediction = output.argmax(1)
                # randomly choose to select the generated word as the next input or the ground truth word.
                probabilities = [self.tf_ratio, 1 - self.tf_ratio]
                idx_choice = np.argmax(np.random.multinomial(1, probabilities))
                if idx_choice == 0:  # choose y[t] as the next word
                    word_t = y[t]
                else:
                    word_t = prediction
        else:
            # encode input sequence
            h_n, c_n = self.encoder(X)
            # set the first input to the SOS
            word_t = y[0]
            for t in range(1, seq_len):
                # compute output, hidden state and cell state
                output, h_n, c_n = self.decoder(word_t, h_n, c_n)
                # update the data structure to hold outputs
                outputs[t] = output
                # take the best prediction from the vocabulary.
                prediction = output.argmax(1)
                probabilities = [self.tf_ratio, 1 - self.tf_ratio]
                idx_choice = np.argmax(np.random.multinomial(1, probabilities))
                # use teaching forcing to randomly chose the next input for the decoder
                if idx_choice == 0:  # choose y[t] as the next word
                    word_t = y[t]
                else:
                    word_t = prediction
        return outputs


class GreedySearch(nn.Module):
    """
    Implements the evaluator that retrieves the best words from the
    vocabulary as the user inputs a phrase.
    """
    def __init__(self, encoder, decoder, voc, attention=True):
        super(GreedySearch, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.voc = voc

    def forward(self, input_seq, max_length=10):
        """
        This function implement the Greedy search technique
        to decode an input sequence of the user and compute the
        most likely sequence of word of the chatbot. In this phase
        we don't use teaching forcing because we want
        to select only the words outputted by the decoder.
        :param input_seq: the input sequence typed by the user
        :param max_length: max length of the input sequence.
        :return predictions: the most likely sequence of words.
        """
        seq_len = input_seq.shape[0]
        outputs = torch.zeros(seq_len, 1, self.voc.__len__()).to(device)
        if self.attention:
            # forward pass the input through the encoder
            encoder_outputs, h_n, c_n = self.encoder(input_seq)
            # set the first input word of the decoder as the '<S>' token.
            word_t = self.voc.word_to_idx['<S>']
            word_t = torch.tensor([word_t]).to(device)
            for t in range(1, seq_len):
                output, h_n, c_n = self.decoder(word_t, h_n, c_n, encoder_outputs)
                outputs[t] = output
                prediction = output.argmax(1)
                word_t = prediction
                word_t = word_t.to(device)
        else:
            h_n, c_n = self.encoder(input_seq)
            # set the first input word of the decoder as the '<S>' token.
            word_t = self.voc.word_to_idx['<S>']
            word_t = torch.tensor([word_t])
            # compute the predictions through the decoder
            for t in range(1, seq_len):
                output, h_n, c_n = self.decoder(word_t, h_n, c_n)
                outputs[t] = output
                prediction = output.argmax(1)
                word_t = prediction
                word_t = word_t.to(device)
        return outputs











