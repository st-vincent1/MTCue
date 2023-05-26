# Author: Sebastian Vincent
# Date: 3 Feb 2023
# MIT License (see repository for details)
#
# Root of model architecture

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture
)

from mtcue.models.mtcue_config import MTCueConfig

from fairseq.models.fairseq_model import check_type
from fairseq.models.transformer import Embedding

from mtcue.modules import (
    ContextEncoder,
    TransformerEncoderFromPretrained,
    TransformerDecoderFromPretrained,
    AugTransformerDecoderFromPretrained,
)
from fairseq.models.transformer.transformer_legacy import base_architecture
from fairseq.models.transformer.transformer_legacy import transformer_vaswani_wmt_en_de_big

import logging
import os


class DoubleEncoderDecoderModel(BaseFairseqModel):
    """Heavily based on FairseqEncoderDecoderModel, with the only changes being the extra encoder.
    Does not implement FairseqEncoderDecoderModel since the signature in forward needs to change"""

    @classmethod
    def build_model(cls, args, task):
        pass

    def __init__(self, cxt_encoder, encoder, decoder):
        super().__init__()

        self.cxt_encoder = cxt_encoder
        self.encoder = encoder
        self.decoder = decoder
        check_type(self.cxt_encoder, FairseqEncoder)
        check_type(self.encoder, FairseqEncoder)
        check_type(self.decoder, FairseqDecoder)

    def forward(self, cxt_vectors, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        raise NotImplementedError

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    # deleted lengths
    def extract_features(self, cxt_vectors, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions(), self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class MTCueTransformerBase(DoubleEncoderDecoderModel):
    """
    Heavily based on fairseq/models/transformer/transformer_base.py:TransformerModelBase
    Adapted to extend DoubleEncoderDecoderModel (implemented above)

    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        cxt_encoder (TransformerEncoder): the context encoder
        encoder (TransformerEncoder): the source encoder
        decoder (TransformerDecoder): the (target) decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    * ``mtcue_{combine_strategy}_qk``: MTCue with {combine_strategy} as the strategy to combine context/source states
    with intermittent decoder states. Available combination strategies include
        * ``flat``: concatenate context/source states with decoder states
        * ``mean``: average context/source states with decoder states
        * ``aug_parallel``: cross-attention of context states with decoder states, and in parallel with source states;
                            then results are added position-wise
        * ``aug_sequential``: cross-attention of decoder states with source encoder states, and then the result's with
                              context encoder states
    """

    def __init__(self, cfg, cxt_encoder, encoder, decoder):
        super().__init__(cxt_encoder, encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = False

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, MTCueConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing

        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.cxt_encoder.layers_to_keep:
            cfg.cxt_encoder.layers = len(cfg.cxt_encoder.layers_to_keep.split(","))
        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )

            if cfg.decoder.embed_path and (
                    cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )

            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        cxt_encoder = cls.build_cxt_encoder(cfg)

        encoder = cls.build_source_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, cxt_encoder, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_cxt_encoder(cls, cfg):
        return ContextEncoder(cfg)

    @classmethod
    def build_source_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderFromPretrained(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        if cfg.context_inclusion == "aug":
            return AugTransformerDecoderFromPretrained(
                cfg, tgt_dict, embed_tokens
            )
        else:
            return TransformerDecoderFromPretrained(
                cfg,
                tgt_dict,
                embed_tokens,
                no_encoder_attn=cfg.no_cross_attention,
            )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            cxt_vectors,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass.
        """
        cxt_encoder_out = self.cxt_encoder(
            cxt_vectors=cxt_vectors,
            return_all_hiddens=return_all_hiddens
        )
        cxt_output_vector = cxt_encoder_out["cxt_encoder_out"][0].transpose(0, 1)
        encoder_out = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths,
            cxt_vector=cxt_output_vector,
            return_all_hiddens=return_all_hiddens
        )

        encoder_out = encoder_out | cxt_encoder_out


        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


@register_model('mtcue_transformer')
class MTCueTransformer(MTCueTransformerBase):
    """MTCue Transformer model. Shadow of TransformerLegacy model."""

    def __init__(self, args, encoder, cxt_encoder, decoder):
        cfg = MTCueConfig.from_namespace(args)
        super().__init__(cfg, encoder, cxt_encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, MTCueConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        cfg = MTCueConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            MTCueConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            MTCueConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            MTCueConfig.from_namespace(args), tgt_dict, embed_tokens
        )


@register_model_architecture('mtcue_transformer', 'mtcue_aug_parallel_qk')
def mtcue_aug_parallel_qk(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'aug')
    args.synthesizer_augmented_cross_attention_merge_type = getattr(
        args, 'synthesizer_augmented_cross_attention_merge_type', 'parallel')
    args.qknorm = getattr(args, 'qknorm', True)
    base_architecture(args)


@register_model_architecture('mtcue_transformer', 'novotney_cue')
def novotney_cue(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'cxt-src-concat')
    args.context_average = getattr(args, 'context_average', True)
    args.qknorm = getattr(args, 'qknorm', True)
    base_architecture(args)

@register_model_architecture('mtcue_transformer', 'abl_pos')
def abl_pos(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'aug')
    args.synthesizer_augmented_cross_attention_merge_type = getattr(
        args, 'synthesizer_augmented_cross_attention_merge_type', 'parallel')
    args.qknorm = getattr(args, 'qknorm', True)
    args.no_doc_positions = getattr(args, 'no_doc_positions', True)
    base_architecture(args)


@register_model_architecture('mtcue_transformer', 'abl_qk')
def abl_qk(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'aug')
    args.synthesizer_augmented_cross_attention_merge_type = getattr(
        args, 'synthesizer_augmented_cross_attention_merge_type', 'parallel')
    base_architecture(args)


@register_model_architecture('mtcue_transformer', 'abl_cxt_enc')
def abl_cxt_enc(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'aug')
    args.synthesizer_augmented_cross_attention_merge_type = getattr(
        args, 'synthesizer_augmented_cross_attention_merge_type', 'parallel')
    args.context_just_embed = getattr(args, 'context_just_embed', True)
    args.tiny_init = getattr(args, 'tiny_init', True)
    args.layernorm_embedding_context = getattr(args, 'layernorm_embedding_context', True)
    base_architecture(args)

@register_model_architecture('mtcue_transformer', 'mtcue_big')
def mtcue_big(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'aug')
    args.synthesizer_augmented_cross_attention_merge_type = getattr(
        args, 'synthesizer_augmented_cross_attention_merge_type', 'parallel')
    args.qknorm = getattr(args, 'qknorm', True)
    args.embed_dim = getattr(args, 'embed_dim', 1024)
    transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture('transformer', 'mtcue_100_pretrain')
def mtcue_pm_pretrain(args):
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    base_architecture(args)

@register_model_architecture('mtcue_transformer', 'mtcue')
def mtcue_aug_parallel_qk(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'aug')
    args.synthesizer_augmented_cross_attention_merge_type = getattr(
        args, 'synthesizer_augmented_cross_attention_merge_type', 'parallel')
    args.qknorm = getattr(args, 'qknorm', True)
    base_architecture(args)

@register_model_architecture('mtcue_transformer', 'tagging_100')
def tagging_pm(args):
    args.context_inclusion = getattr(args, 'context_inclusion', 'cxt-src-concat')
    args.context_just_embed = getattr(args, 'context_just_embed', True)
    args.tiny_init = getattr(args, 'tiny_init', True)
    args.layernorm_embedding_context = getattr(args, 'layernorm_embedding_context', True)
    args.encoder_layers = getattr(args, "encoder_layers", 10)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    base_architecture(args)
