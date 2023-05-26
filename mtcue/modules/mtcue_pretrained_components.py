from mtcue.modules import TransformerEncoderWithContext, TransformerDecoderWithContext, AugTransformerDecoder
from fairseq import checkpoint_utils

import logging


class TransformerEncoderFromPretrained(TransformerEncoderWithContext):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if not self.cfg.load_pretrained_weights or self.cfg.load_dec_only:
            return
        logging.info("Loading checkpoint from pretrained for Encoder.")
        try:
            checkpoint_utils.load_pretrained_component_from_model(
                component=self,
                checkpoint=self.cfg.load_pretrained_weights
            )
        except:
            logging.info("Failed to load pre-trained MT model weights. Unless you're training it's OK")


class TransformerDecoderFromPretrained(TransformerDecoderWithContext):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if not self.cfg.load_pretrained_weights or self.cfg.load_enc_only:
            return
        logging.info("Loading checkpoint from pretrained for Decoder.")
        try:
            checkpoint_utils.load_pretrained_component_from_model(
                component=self,
                checkpoint=self.cfg.load_pretrained_weights
            )
        except:
            logging.info("Failed to load pre-trained MT model weights. Unless you're training it's OK")


class AugTransformerDecoderFromPretrained(AugTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if not self.cfg.load_pretrained_weights or self.cfg.load_enc_only:
            return
        logging.info("Loading checkpoint from pretrained for Decoder.")
        try:
            checkpoint_utils.load_pretrained_component_from_model(
                component=self,
                checkpoint=self.cfg.load_pretrained_weights
            )
        except:
            logging.info("Failed to load pre-trained MT model weights. Unless you're training it's OK")

