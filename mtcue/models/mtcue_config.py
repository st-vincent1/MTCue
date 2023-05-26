# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from fairseq.models.transformer import TransformerConfig
from fairseq.models.transformer.transformer_config import EncDecBaseConfig, DecoderConfig

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

_NAME_PARSER = r"(decoder|encoder|cxt_encoder|quant_noise)_(.*)"


@dataclass
class MTCueConfig(TransformerConfig):
    """Configuration for MTCue model."""
    cxt_encoder: EncDecBaseConfig = EncDecBaseConfig()
    encoder: EncDecBaseConfig = EncDecBaseConfig()
    decoder: DecoderConfig = DecoderConfig()
    dropout: float = field(default=0.3, metadata={"help": "dropout probability"})
    context_dropout: float = field(default=0.25, metadata={"help": "dropout probability for context"})
    context_seq_dropout: bool = field(default=False, metadata={"help": "random dropout for context elements"})
    context_just_embed: bool = field(
        default=False, metadata={"help": "if true, just embed context, don't go through layers of encoder"}
    )
    context_inclusion: str = field(
        default='add-encoder-outputs',
        metadata={"choices": ['cxt-src-concat', 'add-encoder-outputs', 'tag-enc', 'replace-dec-bos'],
                  "help": 'how output from context encoder should be included'}
    )
    context_average: bool = field(
        default=False, metadata={"help": 'average context'}
    )
    embed_dim: int = field(
        default=512, metadata={"help": 'dimension of lin proj out in context encoder'}
    )
    skip_concat: bool = field(
        default=False, metadata={"help": 'add cxt embedding to cxt layer output'}
    )
    load_pretrained_weights: str = field(
        default=None, metadata={"help": 'Load weights for encoder and decoder from a pretrained model.'
                                        'By default loads both encoder and decoder.'}
    )
    load_dec_only: bool = field(
        default=False, metadata={"help": 'Only load pretrained weights for decoder.'}
    )
    load_enc_only: bool = field(
        default=False, metadata={"help": 'Only load pretrained weights for encoder.'}
    )
    context_layer: int = field(
        default=6, metadata={"help": 'count of layers to merge context and source'}
    )
    no_doc_context: bool = field(
        default=False, metadata={"help": 'dont use document context'}
    )
    doc_context_len: int = field(
        default=6, metadata={"help": 'length of document context'}
    )
    no_meta_context: bool = field(
        default=False, metadata={"help": 'dont use metadata context'}
    )
    no_doc_positions: bool = field(
        default=False, metadata={"help": 'dont embed document positions; if True, treats doc sentences like other contexts'}
    )
    load_pretrained_strict: bool = field(
        default=True, metadata={"help": "Load pretrained weights strictly? Set to True by default but changed to False"
                                        "for decoder_aug since this decoder adds extra components that don't exist in"
                                        "original Transformer "}
    )
    layernorm_embedding_context: bool = field(
        default=False, metadata={"help": "Use layernorm on context embeddings"}
    )
    qknorm: bool = field(
        default=True, metadata={"help": 'use QKnorm instead of dot product'}
    )
    qknorm_scale: float = field(
        default=8.41, metadata={"help": "Scale QK by this value; this is only for QK norm"}
    )
    tiny_init: bool = field(
        default=False, metadata={"help": "Use tiny init for context encoder"}
    )
    init_pos_to_0: bool = field(
        default=False, metadata={"help": "Initialise position embedding for doc context to 0s (like in tinyemb)"}
    )
