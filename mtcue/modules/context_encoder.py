# Author: Sebastian Vincent
# Date: 3 Feb 2023
# MIT License (see repository for details)
#
# Context Encoder of MTCue

import math
from typing import Dict, List, Optional, Tuple
from random import randint

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from mtcue.models import MTCueConfig
from mtcue.modules import context_layer
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from einops import rearrange
import logging


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "ContextEncoderBase":
        return "ContextEncoder"
    else:
        return module_name


class ContextEncoderBase(FairseqEncoder):
    """
    Encoder for context.

    Transformer cxt_encoder consisting of *cfg.cxt_encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary=None)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.context_dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.context_seq_dropout = cfg.context_seq_dropout
        self.no_doc_context = cfg.no_doc_context
        self.no_meta_context = cfg.no_meta_context
        self.doc_context_len = cfg.doc_context_len
        self.no_doc_positions = cfg.no_doc_positions
        self.return_fc = return_fc

        self.cxt_embed_dim = 384 if cfg.context_model in ["minilm", "hash", "random"] else 768
        self.embed_dim = cfg.embed_dim
        self.max_source_positions = cfg.max_source_positions
        self.padding_idx = 0

        # Make a sequential embedding layer with layernorm afterwards
        self.lin_proj = nn.Linear(self.cxt_embed_dim, self.embed_dim, bias=False)
        if cfg.tiny_init:
            # Initiating lin proj to tiny values as per https://github.com/BlinkDL/SmallInitEmb
            nn.init.uniform_(self.lin_proj.weight, a=-1e-3, b=1e-3)  # SmallInit(Emb)
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(self.embed_dim)

        if not self.no_doc_positions:
            # Build positional embedding for document context
            # 0 corresponds to current sentence, 1 to previous sentence etc
            self.doc_pos_embed = nn.Embedding(self.doc_context_len, self.embed_dim)

            if cfg.tiny_init or cfg.init_pos_to_0:
                # Setting to 0 to work well with tiny embeddings
                nn.init.constant_(self.doc_pos_embed.weight, 0.0)
            if cfg.tiny_init:
                nn.init.constant_(self.doc_pos_embed.weight, 0.0)

        if cfg.layernorm_embedding_context:
            self.layernorm_embedding = LayerNorm(self.embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if not self.cfg.context_just_embed:
            self.layers = nn.ModuleList([])
            self.layers.extend(
                [self.build_cxt_encoder_layer(cfg) for i in range(cfg.context_layer)]
            )
            self.num_layers = len(self.layers)
            if cfg.cxt_encoder.normalize_before:
                self.layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
            else:
                self.layer_norm = None
        else:
            self.num_layers = 0
            self.layer_norm = None



    def build_cxt_encoder_layer(self, cfg):
        layer = context_layer.ContextEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(self, contexts: Tuple[Optional[Tensor], Optional[Tensor]]
                          ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            contexts: Tuple of (document_vectors, metadata_vectors)
                document_vectors: (bsz, #sentences, 768)
                metadata_vectors: (bsz, #sentences, 768)

        """
        # Resolving document and metadata context. Adds pos_embed to doc context; resolves context into a single entity
        document, metadata = contexts
        assert document is not None or metadata is not None
        bsz = document.shape[0] if document is not None else metadata.shape[0]
        device = document.device if document is not None else metadata.device
        context_embedding = torch.empty(bsz, 0, self.embed_dim).to(device)
        if self.cfg.fp16:
            context_embedding = context_embedding.half()
        context_vectors = torch.empty(bsz, 0, self.cxt_embed_dim).to(device)
        if self.cfg.fp16:
            context_vectors = context_vectors.half()
        if document is not None:
            document_embedding = self.lin_proj(document)
            # (bsz, #sentences, embed_dim)
            if not self.no_doc_positions:
                # Create a position vector to add to document embeddings
                markers_d = self.doc_pos_embed(torch.arange(document.shape[1], device=document.device))
                document_embedding += markers_d
            context_embedding = torch.cat((context_embedding, document_embedding), dim=1)
            context_vectors = torch.cat((context_vectors, document), dim=1)

        if metadata is not None:
            metadata_embedding = self.lin_proj(metadata)
            context_embedding = torch.cat((context_embedding, metadata_embedding), dim=1)
            context_vectors = torch.cat((context_vectors, metadata), dim=1)

        x = self.embed_scale * context_embedding

        non_empty_mask = context_vectors.abs().sum(dim=2).bool()

        non_empty_rows = non_empty_mask.any(dim=0).nonzero(as_tuple=False).squeeze(1)
        x = x[:, non_empty_rows, :]  # (bsz, cxt_len, 768)

        context_vectors = context_vectors[:, non_empty_rows, :]  # (bsz, cxt_len, 768)
        # Code adapted from
        # "https://github.com/lucidrains/perceiver-ar-pytorch/blob/main/perceiver_ar_pytorch/perceiver_ar_pytorch.py"
        if self.training and self.context_seq_dropout and context_embedding.size(1) > 0:
            cxt_len = x.size(1)
            keep_context_len = randint(1, cxt_len)  # Keep at least half of the contexts

            # Drop out some contexts to improve generalization
            rand = torch.zeros(bsz, cxt_len).uniform_()
            keep_indices = rand.topk(keep_context_len, dim=-1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()
            x = rearrange(x[keep_mask], '(b n) d -> b n d', b=bsz)

            context_vectors = rearrange(context_vectors[keep_mask], '(b n) d -> b n d', b=bsz)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, context_vectors

    def forward(self, cxt_vectors, return_all_hiddens: bool = False):
        return self.forward_scriptable(
            cxt_vectors, return_all_hiddens
        )

    def forward_scriptable(self, cxt_vectors, return_all_hiddens: bool = False):
        """
        Args:
            cxt_vectors (LongTensor): context vectors in the shape of
                `(batch, cxt_len, cxt_embed_dim)`
            return_all_hiddens (bool, optional): also return all the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **cxt_encoder_out** (Tensor): the last cxt_encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **cxt_encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
                - **cxt_encoder_attn** (List[Tensor]): all intermediate
                    attention weights of shape `(batch, num_heads, src_len, src_len)`.
                    Only populated if *return_all_hiddens* is True.
                - **cxt_encoder_padding_mask** (Tensor): the cxt_encoder padding mask of
                    shape `(batch, src_len)`
                - **cxt_lengths** (Tensor): the cxt_lengths of shape `(batch)`
        """
        x, cxt_vectors = self.forward_embedding(cxt_vectors)
        encoder_padding_mask = torch.mean(cxt_vectors, dim=2).eq(self.padding_idx)

        has_pads = cxt_vectors.device.type == "xla" or encoder_padding_mask.any()
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        cxt_encoder_states = []
        fc_results = []

        if return_all_hiddens:
            cxt_encoder_states.append(x)

        # if we are not just using the lin proj. context embeddings
        if not self.cfg.context_just_embed and x.nelement() > 0:
            for layer in self.layers:
                lr = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)

                if isinstance(lr, tuple) and len(lr) == 2:
                    x, fc_result = lr
                else:
                    x = lr
                    fc_result = None

                if return_all_hiddens and not torch.jit.is_scripting():
                    assert cxt_encoder_states is not None
                    cxt_encoder_states.append(x)
                    fc_results.append(fc_result)

            if self.layer_norm is not None:
                x = self.layer_norm(x)

        # Lengths of cxt vectors (not including cls token)
        cxt_lengths = (
            torch.mean(cxt_vectors, dim=2).ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )

        # Average x across sequence dimension, don't take into account padded out tokens, and watch out for div by 0
        if self.cfg.context_average:
            x = torch.sum(x, dim=0) / torch.max(cxt_lengths, torch.ones_like(cxt_lengths))
            x = x.unsqueeze(0)

        return {
            "cxt_encoder_out": [x],  # T x B x C
            "cxt_encoder_states": cxt_encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "cxt_vectors": [],  # B x T x E
            "cxt_lengths": [cxt_lengths],  # B x 1
            "cxt_encoder_padding_mask": [encoder_padding_mask],  # B x T
        }

    @torch.jit.export
    def reorder_encoder_out(self, cxt_encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder cxt_encoder output according to *new_order*.

        Args:
            cxt_encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *cxt_encoder_out* rearranged according to *new_order*
        """
        if len(cxt_encoder_out["cxt_encoder_out"]) == 0:
            new_cxt_encoder_out = []
        else:
            new_cxt_encoder_out = [cxt_encoder_out["cxt_encoder_out"][0].index_select(1, new_order)]
        if len(cxt_encoder_out["cxt_encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                cxt_encoder_out["cxt_encoder_padding_mask"][0].index_select(0, new_order)
            ]

        if len(cxt_encoder_out["cxt_vectors"]) == 0:
            cxt_vectors = []
        else:
            cxt_vectors = [(cxt_encoder_out["cxt_vectors"][0]).index_select(0, new_order)]
        if len(cxt_encoder_out["cxt_lengths"]) == 0:
            new_cxt_lengths = []
        else:
            new_cxt_lengths = [
                cxt_encoder_out["cxt_lengths"][0].index_select(0, new_order)
            ]
        cxt_encoder_states = cxt_encoder_out["cxt_encoder_states"]
        if len(cxt_encoder_states) > 0:
            for idx, state in enumerate(cxt_encoder_states):
                cxt_encoder_states[idx] = state.index_select(1, new_order)

        return {
            "cxt_encoder_out": new_cxt_encoder_out,  # T x B x C
            "cxt_encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "cxt_encoder_states": cxt_encoder_states,  # List[T x B x C]
            "cxt_vectors": cxt_vectors,  # B x T
            "cxt_lengths": new_cxt_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, cxt_encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return cxt_encoder_out

    def max_positions(self):
        """Maximum input length supported by the cxt_encoder."""
        return 1024

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class ContextEncoder(ContextEncoderBase):
    def __init__(self, args, return_fc=False):
        self.args = args
        super().__init__(
            MTCueConfig.from_namespace(args),
            return_fc=return_fc,
        )

    def build_cxt_encoder_layer(self, args):
        return super().build_cxt_encoder_layer(
            MTCueConfig.from_namespace(args),
        )
