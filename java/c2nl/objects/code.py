from c2nl.inputters.vocabulary import Vocabulary, BOS_WORD, EOS_WORD


class Code(object):
    """
    Code containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._language = None
        self._text = None
        self._tokens = []
        self._type = []
        self._subtoken = []
        self._intoken = []
        self._instatement = []
        self._mask = []
        self._dataflow = None
        self._controlflow = None
        self.src_vocab = None  # required for Copy Attention

    @property
    def id(self) -> str:
        return self._id

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, param: str) -> None:
        self._language = param

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def type(self) -> list:
        return self._type

    @type.setter
    def type(self, param: list) -> None:
        assert isinstance(param, list)
        self._type = param

    @property
    def keyword(self) -> list:
        return self._keyword

    @keyword.setter
    def keyword(self, param: list) -> None:
        assert isinstance(param, list)
        self._keyword = param

    @property
    def type5(self) -> list:
        return self._type5

    @type5.setter
    def type5(self, param: list) -> None:
        assert isinstance(param, list)
        self._type5 = param

    @property
    def intoken(self) -> list:
        return self._intoken

    @intoken.setter
    def intoken(self, param: list) -> None:
        assert isinstance(param, list)
        self._intoken = param

    @property
    def subtoken(self) -> list:
        return self._subtoken

    @subtoken.setter
    def subtoken(self, param: list) -> None:
        assert isinstance(param, list)
        self._subtoken = param
    
    @property
    def instatement(self) -> list:
        return self._instatement

    @instatement.setter
    def instatement(self, param: list) -> None:
        assert isinstance(param, list)
        self._instatement = param

    @property
    def dataflow(self) -> list:
        return self._dataflow

    @dataflow.setter
    def dataflow(self, param) -> None:
        # assert isinstance(param, list)
        self._dataflow = param

    @property
    def controlflow(self) -> list:
        return self._controlflow

    @controlflow.setter
    def controlflow(self, param) -> None:
        # assert isinstance(param, list)
        self._controlflow = param

    @property
    def mask(self) -> list:
        return self._mask

    @mask.setter
    def mask(self, param: list) -> None:
        assert isinstance(param, list)
        self._mask = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param
        self.form_src_vocab()

    def form_src_vocab(self) -> None:
        self.src_vocab = Vocabulary()
        assert self.src_vocab.remove(BOS_WORD)
        assert self.src_vocab.remove(EOS_WORD)
        self.src_vocab.add_tokens(self.tokens)

    def vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False
