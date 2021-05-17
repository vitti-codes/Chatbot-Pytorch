import torch
from torch.utils.data import Dataset


class CornellCorpus(Dataset):
    """
    Implements the functionality of the dataset. Inheriths __getitem__ and __len__ from
    Pytorch's Dataset class. The first inherited function is used to retrieve a datapoint in
    the dataset at a given index.
    """

    def __init__(self, dialogs, vocabulary, stage='train', split_ratio=[0.8, 0.1, 0.1], max_length=10):
        super(CornellCorpus, self).__init__()
        self.vocabulary = vocabulary
        self.dialogs_pair_idx = dialogs
        self.max_length = max_length
        self.split_ratio = split_ratio
        self.stage = stage
        # build the dataset splitting the data into training, validation and testing
        self.data = self.build_data()

    def build_data(self):
        """
        This function builds the dataset by splitting the data into
        training, validation and testing sets.
        :return:
        """
        # define limit index for each splitting set
        train_limit = int(len(self.dialogs_pair_idx)*self.split_ratio[0])
        val_limit = train_limit + int(len(self.dialogs_pair_idx)*self.split_ratio[1])
        test_limit = int(len(self.dialogs_pair_idx))
        if self.stage == 'train':
            return self.choose_pairs(0, train_limit)
        elif self.stage == 'val':
            return self.choose_pairs(train_limit, val_limit)
        elif self.stage == 'test':
            return self.choose_pairs(val_limit, test_limit)

    def choose_pairs(self, start_limit, end_limit):
        """
        Splits the data and builds a set from a given range.
        :param start_limit: the start point by which data are selected
        :param end_limit: the ending point by which data are selected
        :return: the set of selected data
        """
        dataset = []
        for dialog in self.dialogs_pair_idx[start_limit: end_limit]:
            q_a_pair = [self.vocabulary.idx_to_text[dialog[0]], self.vocabulary.idx_to_text[dialog[1]]]
            if q_a_pair[0] != ' ' and q_a_pair[1] != ' ':
                dataset.append(q_a_pair)
        return dataset

    def pad_sequence(self, sequence):
        """
        Insert at the end of the sentence the EOS token and pads the
        sentence if needed.
        :param sequence: the sequence to be processed. The sequence is a list of word indices
        :return:
        """
        # useful tokens
        pad_token_idx = self.vocabulary.word_to_idx['<PAD>']
        end_token_idx = self.vocabulary.word_to_idx['</S>']
        start_token_idx = self.vocabulary.word_to_idx['<S>']
        # append end token
        sequence.append(end_token_idx)
        #pad the rest of the characters
        while len(sequence) <= self.max_length:
            sequence.append(pad_token_idx)
        # insert start token
        sequence.insert(0, start_token_idx)
        return sequence

    def check_length_requirement(self, sequence):
        """
        Either pads or trunk the sentence and append EOS.
        :param sequence: the sequence to be processed
        :return:
        """
        if len(sequence) < self.max_length:
            # pad sequence and append EOS and S
            return self.pad_sequence(sequence)
        elif len(sequence) > self.max_length:
            # trunk sequence and append EOS and S
            sequence = sequence[:self.max_length]
            end_token_idx = self.vocabulary.word_to_idx['</S>']
            start_token_idx = self.vocabulary.word_to_idx['<S>']
            sequence.append(end_token_idx)
            sequence.insert(0, start_token_idx)
            return sequence
        else:
            end_token_idx = self.vocabulary.word_to_idx['</S>']
            start_token_idx = self.vocabulary.word_to_idx['<S>']
            sequence.append(end_token_idx)
            sequence.insert(0, start_token_idx)
            return sequence

    def process_batch(self, batch):
        """
        This function takes a batch and for both query and answer
        converts each word to the relative index in the vocabulary, pads
        the sentences that are shorter than 'max_length' and append the
        <EOS> token. The <S> is inserted at the beginning of the sentence.
        Finally the sentences are converted to tensors.
        :param batch: the batch composed by a pair of question/answer
        :return: the batch containing the pair of tensor relatives to the question/answer.
        """
        question = batch[0]
        answer = batch[1]
        question_idx = []
        answer_idx = []
        for q_word in question.strip().split(" "):
            # fill the list with the corresponding index2word mapping for the question
            question_idx.append(self.vocabulary.word_to_idx[q_word])
        for a_word in answer.strip().split(" "):
            # fill the list with the corresponding index2word mapping for the answer
            answer_idx.append(self.vocabulary.word_to_idx[a_word])

        # check if either or both the pair has a length greater or smaller than max_length
        question_idx = self.check_length_requirement(question_idx)
        answer_idx = self.check_length_requirement(answer_idx)
        # convert to tensors
        question_idx = torch.tensor(question_idx)
        answer_idx = torch.tensor(answer_idx)
        return [question_idx, answer_idx]

    def __getitem__(self, idx):
        return self.process_batch(self.data[idx])

    def __len__(self):
        return len(self.data)
