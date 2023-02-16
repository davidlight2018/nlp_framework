from transformers import BertTokenizer, AutoTokenizer, BertTokenizerFast, PreTrainedTokenizer


class CNerTokenizer(BertTokenizer):
    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens
