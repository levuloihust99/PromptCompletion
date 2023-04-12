import logging
from typing import Text, List
import sentencepiece as spm

logger = logging.getLogger(__name__)


class SPieceNFKCTokenizer(object):
    def __init__(
        self,
        spiece_model_path: Text,
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>"
    ):
        spiece_model = spm.SentencePieceProcessor()
        spiece_model.LoadFromFile(spiece_model_path)

        # CLS token
        cls_token_id = spiece_model.PieceToId(cls_token)
        if cls_token_id == spiece_model.unk_id():
            raise Exception("Invalid cls_token: '{}' is not in the vocabulary.".format(cls_token))
        self.cls_token = cls_token
        self.cls_token_id = cls_token_id

        # SEP token
        sep_token_id = spiece_model.PieceToId(sep_token)
        if sep_token_id == spiece_model.unk_id():
            raise Exception("Invalid sep_token: '{}' is not in the vocabulary.".format(sep_token))
        self.sep_token = sep_token
        self.sep_token_id = sep_token_id

        # MASK token
        mask_token_id = spiece_model.PieceToId(mask_token)
        if mask_token_id == spiece_model.unk_id():
            raise Exception("Invalid mask_token: '{}' is not in the vocabulary.".format(mask_token))
        self.mask_token = mask_token
        self.mask_token_id = mask_token_id

        self._spiece_tokenizer = spiece_model
    
    def tokenize(self, text: Text):
        return self._spiece_tokenizer.EncodeAsPieces(text)
    
    def encode(self, text: Text):
        return self._spiece_tokenizer.EncodeAsIds(text)
    
    def convert_ids_to_tokens(self, token_ids: List[int]):
        return [self._spiece_tokenizer.IdToPiece(tok_id) for tok_id in token_ids]
    
    def convert_tokens_to_ids(self, tokens: List[Text]):
        return [self._spiece_tokenizer.PieceToId(tok) for tok in tokens]
    
    def convert_token_to_id(self, token: Text):
        return self._spiece_tokenizer.PieceToId(token)
    
    def convert_id_to_token(self, token_id: int):
        return self._spiece_tokenizer.IdToPiece(token_id)

    @property
    def vocab_size(self):
        return self._spiece_tokenizer.vocab_size()

    @property
    def pad_token_id(self):
        return self._spiece_tokenizer.pad_id()

    @property
    def pad_token(self):
        return self._spiece_tokenizer.IdToPiece(
            self._spiece_tokenizer.pad_id())

    @property
    def unk_token_id(self):
        return self._spiece_tokenizer.unk_id()
    
    @property
    def unk_token(self):
        return self._spiece_tokenizer.IdToPiece(
            self._spiece_tokenizer.unk_id())

    @property
    def bos_token_id(self):
        return self._spiece_tokenizer.bos_id()

    @property
    def bos_token(self):
        return self._spiece_tokenizer.IdToPiece(
            self._spiece_tokenizer.bos_id())

    @property
    def eos_token_id(self):
        return self._spiece_tokenizer.eos_id()
    
    @property
    def eos_token(self):
        return self._spiece_tokenizer.IdToPiece(
            self._spiece_tokenizer.eos_id())
