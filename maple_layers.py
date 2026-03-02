"""
MaPLe (Low-Rank Adaptation) implementation for SAM3 model fine-tuning.
Supports selective application to different transformer components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Callable, Tuple, Union
import math

from sam3.train.data.sam3_image_dataset import InferenceMetadata
from sam3.model.text_encoder_ve import text_global_pool

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in the model.

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def save_maple_weights(model: nn.Module, save_path: str):
    """
    Save only MaPLe weights (not the full model).

    Args:
        model: Model with MaPLe layers
        save_path: Path to save MaPLe weights
    """
    language_backbone = model.backbone.language_backbone
    maple_inputs_embeds = torch.cat([language_backbone.embedding[:, :1, :], 
               language_backbone.prefix_embedding, 
               language_backbone.suffix_embedding, 
               language_backbone.embedding[:, language_backbone.prefix_embedding.shape[1]+language_backbone.suffix_embedding.shape[1]+1:, :]], dim=1)
    maple_state_dict = {
        "maple_inputs_embeds": maple_inputs_embeds
    }

    torch.save(maple_state_dict, save_path)
    print(f"Saved MaPLe weights to {save_path}")


def load_maple_weights(load_path: str):
    """
    Load MaPLe weights into a model.

    Args:
        model: Model with MaPLe layers
        load_path: Path to MaPLe weights
    """
    maple_state_dict = torch.load(load_path)
    print(f"Loaded MaPLe weights from {load_path}")
    return maple_state_dict


@dataclass
class FindQuery:
    query_text: Union[str, dict]

    image_id: int

    # In case of a find query, the list of object ids that have to be predicted
    object_ids_output: List[int]

    # This is "instance exhaustivity".
    # true iff all instances are separable and annotated
    # See below the slightly different "pixel exhaustivity"
    is_exhaustive: bool
    
    # The order in which the queries are processed (only meaningful for video)
    query_processing_order: int = 0

    # Input geometry, initially in denormalized XYXY format. Then
    # 1. converted to normalized CxCyWH by the Normalize transform
    input_bbox: Optional[torch.Tensor] = None
    input_bbox_label: Optional[torch.Tensor] = None

    # Only for the PVS task
    input_points: Optional[torch.Tensor] = None

    semantic_target: Optional[torch.Tensor] = None

    # pixel exhaustivity: true iff the union of all segments (including crowds)
    # covers every pixel belonging to the target class
    # Note that instance_exhaustive implies pixel_exhaustive
    is_pixel_exhaustive: Optional[bool] = None


@dataclass
class FindQueryLoaded(FindQuery):
    # Must have default value since FindQuery has entries with default values
    inference_metadata: Optional[InferenceMetadata] = None


class TextTransformer(nn.Module):
    def __init__(
        self,
        model,
        no_causal_mask: bool = False,
        proj_bias: bool = False,
    ):
        super().__init__()
        assert model.pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = model.output_tokens
        self.num_pos = self.context_length = model.context_length
        self.vocab_size = model.vocab_size
        self.width = model.width
        self.output_dim = model.output_dim
        self.heads = model.heads
        self.pool_type = model.pool_type

        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )
        if proj_bias:
            self.text_projection = model.text_projection
        else:
            self.text_projection = model.text_projection

    def build_causal_mask(self) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, text: torch.Tensor, x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_len = text.shape[1]

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled
    

class VETextEncoder(nn.Module):
    def __init__(
        self,
        model,
        device,
        config=None
    ):
        super().__init__()
        self.context_length = model.context_length
        self.use_ln_post = model.use_ln_post
        self.tokenizer = model.tokenizer
        self.config = config

        self.encoder = TextTransformer(model.encoder)
        self.resizer = model.resizer
    
        if self.config is not None:
            with torch.no_grad():
                self.tokenized_prompts = self.tokenizer(f'{self.config["prefix"]["prompt"]} {self.config["suffix"]["prompt"]}', context_length=self.context_length).to(device)
                self.embedding = self.encoder.token_embedding(self.tokenized_prompts)

                prefix_prompts = self.tokenizer({self.config["prefix"]["prompt"]}, context_length=self.context_length).to(device)
                suffix_prompts = self.tokenizer({self.config["suffix"]["prompt"]}, context_length=self.context_length).to(device)

                self.prefix_embedding, self.suffix_embedding = None, None
                non_zero_quantity = (prefix_prompts > 0).sum().item()
                self.prefix_embedding = self.encoder.token_embedding(prefix_prompts)[:, 1:non_zero_quantity-1]
                self.prefix_seq_len = self.prefix_embedding.shape[1]

                non_zero_quantity = (suffix_prompts > 0).sum().item()
                self.suffix_embedding = self.encoder.token_embedding(suffix_prompts)[:, 1:non_zero_quantity-1]
                self.suffix_seq_len = self.suffix_embedding.shape[1]

                if self.config["prefix"]["tuning"]:
                    self.prefix_embedding = nn.Parameter(self.prefix_embedding)
                if self.config["suffix"]["tuning"]:
                    self.suffix_embedding = nn.Parameter(self.suffix_embedding)
                    
                assert self.config["prefix"]["tuning"] or self.config["suffix"]["tuning"], "not supported"

            self.text_attention_mask = (self.tokenized_prompts != 0).bool()
            self.text_attention_mask = self.text_attention_mask.ne(1).cuda()

    def forward(
        self,
        text: Optional[Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict], List[dict]]]=[],
        input_boxes: Optional[List] = None,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # import pdb; pdb.set_trace()
        if text[0]:
            if isinstance(text[0], dict):
                tokenized_prompts = self.tokenizer(text[0]["text"], context_length=self.context_length).to(device)
                text_attention_mask = (tokenized_prompts != 0).bool()
                text_attention_mask = text_attention_mask.ne(1).cuda()

                inputs_embeds = text[0]["maple_inputs_embeds"]
                
                _, text_memory = self.encoder(tokenized_prompts, inputs_embeds)  # [b, seq_len, d=1024]
                text_memory = text_memory.transpose(0, 1)
                text_memory_resized = self.resizer(text_memory)
            elif isinstance(text[0], str):
                # no use case for this
                assert input_boxes is None or len(input_boxes) == 0, "not supported"

                # Encode the text
                tokenized = self.tokenizer(text, context_length=self.context_length).to(
                    device
                )  # [b, seq_len]
                text_attention_mask = (tokenized != 0).bool()

                # manually embed the tokens
                inputs_embeds = self.encoder.token_embedding(
                    tokenized
                )  # [b, seq_len, d=1024]
                _, text_memory = self.encoder(tokenized)  # [b, seq_len, d=1024]

                assert text_memory.shape[1] == inputs_embeds.shape[1]
                # Invert attention mask because its the opposite in pytorch transformer
                text_attention_mask = text_attention_mask.ne(1)
                # Transpose memory because pytorch's attention expects sequence first
                text_memory = text_memory.transpose(0, 1)
                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)
            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text
                inputs_embeds = tokenized["inputs_embeds"]
                assert input_boxes is None or len(input_boxes) == 0, (
                    "Can't replace boxes in text if it's already encoded"
                )
        else:
            # no use case for this
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            if self.config["prefix"]["tuning"] and self.config["suffix"]["tuning"]:
                inputs_embeds = torch.cat([self.embedding[:, :1, :], self.prefix_embedding, self.suffix_embedding, self.embedding[:, self.prefix_seq_len+self.suffix_seq_len+1:, :]], dim=1)
            elif self.config["prefix"]["tuning"]:
                inputs_embeds = torch.cat([self.embedding[:, :1, :], self.prefix_embedding, self.embedding[:, self.prefix_seq_len+1:, :]], dim=1)
            elif self.config["suffix"]["tuning"]:
                inputs_embeds = torch.cat([self.embedding[:, :self.prefix_seq_len+1, :], self.suffix_embedding, self.embedding[:, self.prefix_seq_len+self.suffix_seq_len+1:, :]], dim=1)

            _, text_memory = self.encoder(self.tokenized_prompts, inputs_embeds)  # [b, seq_len, d=1024]
            text_attention_mask = self.text_attention_mask
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = self.resizer(text_memory)
            
        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )

def _create_text_encoder(model: nn.Module, device, config) -> VETextEncoder:
    """Create SAM3 text encoder."""
    return VETextEncoder(model, device, config=config)

def apply_maple_to_model(model: nn.Module, device, config) -> nn.Module:
    """
    Apply MaPLe to specified modules in the SAM3 model.

    Args:
        model: SAM3 model to apply MaPLe to
        config: MaPLe configuration

    Returns:
        Model with MaPLe applied
    """

    # CRITICAL: Freeze all base model parameters first
    # for param in model.parameters():
    #     param.requires_grad = False
    model.backbone.language_backbone = _create_text_encoder(model.backbone.language_backbone, device, config)
    for name, param in model.named_parameters():
        if "prefix_embedding" in name and config["prefix"]["tuning"] or "suffix_embedding" in name and config["suffix"]["tuning"]:
            print(f"Fine tuning block {name}")
        else:
            param.requires_grad = False
    return model

