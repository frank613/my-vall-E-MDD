"""
Core model for handling all VALL-E tasks.
This should handle all the "low" level things such as:
* parsing inputs to sequences
* converting sequences to embeddings
* forward pass
* processing loss and returning logits

Additional functionality (preparing inputs, generating full audio) should be delegated to classes that inheret the base model
"""

# to-do: clean this whole mess up

import math
import torch
import torch.nn.functional as F
import random
import numpy as np
import re

from time import perf_counter
from collections import namedtuple
from typing import Literal, overload, Optional, Tuple
from functools import partial
from einops import rearrange

from torch import Tensor, einsum, nn
from torch.nn import Embedding
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, MulticlassPrecision

from .arch import *
from ..utils import ml, clamp
from ..samplers import *

# yuck, kind of needed
from ..data import get_task_symmap

import logging

_logger = logging.getLogger(__name__)

# these seem more elegant than a dict
Logits = namedtuple('Logits', ['logits', 'state', 'inputs', 'loss', 'attentions', 'hidden_states'])
Sampled = namedtuple('Sampled', ['ids', 'logits', 'scores', 'entropy'])
LossStats = namedtuple('LossStats', ['loss', 'stats'])

summed_embeddings_task = [ "stt" ]
special_tasks = [ "len", "stt", "phn", "text", "un-phn" ]
non_tokened_names = ["task", "dropout_mask", "classifier_level"]
task_outputs = {
	"tts": "resp",
	"ns": "resp",
	"sr": "resp",
	"stt": "phn",
	"len": "len",
	"phn": "phn",
	"un-phn": "text",
}

# yuck
def _get_offsets(): ##all the tokens for the unified(un-splitted) classifier?
	return {
		"phn": (0, 256), 
		"quant_level": (256, 264), 
		"lang": (264, 270), 
		"task": (270, 279), 
		"len": (279, 290), 
		"tone": (290, 291), 
		"sep": (291, 292), 
		"prom|0": (292, 1316), 
		"prom|1": (1316, 2340), 
		"prom|2": (2340, 3364), 
		"prom|3": (3364, 4388), 
		"prom|4": (4388, 5412), 
		"prom|5": (5412, 6436), 
		"prom|6": (6436, 7460), 
		"prom|7": (7460, 8484), 
		"resps|AR:0:0": (8484, 9509), 
		"resps|NAR:0:1": (9509, 10533), 
		"resps|NAR:1:2": (10533, 11557), 
		"resps|NAR:2:3": (11557, 12581), 
		"resps|NAR:3:4": (12581, 13605), 
		"resps|NAR:4:5": (13605, 14629), 
		"resps|NAR:5:6": (14629, 15653), 
		"resps|NAR:6:7": (15653, 16677), 
		"resps|NAR:0:0": (16677, 17702), 
	}

def _dropout_mask( input, p ):
	return (torch.rand(input.shape[0], device=input.device) < p)

def _create_mask(l, device):
	"""1 is valid region and 0 is invalid."""
	seq = torch.arange(max(l), device=device).unsqueeze(0)  # (1 t)
	stop = torch.tensor(l, device=device).unsqueeze(1)  # (b 1)
	return (seq < stop).float()  # (b t)

def _join(x: tuple[Tensor], sep: Tensor):
	"""
	Args:
		x: (k t d)
		sep: (d)
	"""
	ret = x[0]
	for i in range(1, len(x)):
		ret = torch.cat((ret, sep[None], x[i]), dim=0)
	return ret

def list_to_tensor(x_list: list[Tensor]):
	l = list(map(len, x_list))
	x = pad_sequence(x_list, batch_first=True)
	m = _create_mask(l, x_list[0].device)

	m = m.to(x).int()
	return x, m

def _interleave_sequence_reshape( input: list[torch.Tensor], dim=-1 ):
	shape = (input[0].shape[0] * len(input), input[0].shape[dim] )
	return torch.concat( [ i.t() for i in input ] ).t().reshape( shape )

def _interleave_sequence_flatten( input: list[torch.Tensor] ):
	return torch.concat( [ i.t() for i in input ] ).t().flatten()

# Embedding that sums each codebook level within a given input acoustic prompt
# Mostly to handle some oversights and errors during testing
class AudioEmbedding(nn.Module):
	def __init__(
		self,
		l_embedding_tokens: list[int], # list of number of tokens (needed because AR resps includes stop token)
		token_dim: int, # dimensionality of the embedding
		sums: bool = True, # whether to sum all previous layers of embeddings to factor in other codebook levels (I do not know which way is better)
		l_embedding_names: list[str] = [], # names to map to indices
	):
		super().__init__()
		# array of embeddings
		#   proms are [0, resp_levels]
		#   resp are split to where [0] is for the AR, and [1:] are reserved for NAR (except [-1] for NAR-len if utilized)
		self.embeddings = nn.ModuleList([ml.Embedding(n_tokens, token_dim) for n_tokens in l_embedding_tokens])
		# further experimentation is needed to see if this actually is useful
		self.sums = sums
		# index of name maps to its corresponding embedding in the list
		self.names = l_embedding_names

	def forward(
		self,
		xi: Tensor, # input tensor
		offset: int | None = None, # explicit offset, interop for the older codebase. use `name` instead
		quant_level: int | None = None, # the codebook level of the audio we currently have (our `input_quant_level`)
		name: str | None = None, # specifies where in the embeddings list to start from and iterate through
		sums = None 
	) -> Tensor:
		# if not explicitly requested, use the default setting at instantiation time
		if sums is None:
			sums = self.sums
		
		# if not explicitly requested, assume input quant_level based on shape
		if quant_level is None:
			quant_level = 0 if xi.dim() == 1 else xi.shape[-1] - 1

		# handle mapping embedding index offset
		if name in self.names:
			offset = self.names.index( name )
			offset -= quant_level # offset by quant_level since it'll iterate up that many levels
		
		# sum all prior codebook levels if requested (as quant_level = 0 does not have any other codebooks to sum through)
		if sums and quant_level > 0:
			x = sum( [ self.embeddings[input_quant_level + offset]( xi[:, input_quant_level] ) for input_quant_level in range( quant_level + 1 ) ] )
		else:
			input_quant_level = quant_level
			x = self.embeddings[input_quant_level + offset]( xi if xi.dim() == 1 else xi[:, input_quant_level] )

		return x

# per-level classification
# it might actually be "better" in the long run to only have one output head like a traditional LM, and just de-stitch it here instead of doing modulus math and whatever like the HF/experimental impl
class Classifiers(nn.Module):
	def __init__(
		self,
		l_embedding_tokens: list[int], # list of number of tokens (needed because AR resps includes stop token)
		l_embedding_names: list[str], # list of names to map to each classifier,
		d_model: int, # dimensionality of the embedding
		bias: bool = True,
	):
		super().__init__()
		self.proj = nn.ModuleList([nn.Linear(d_model, n_tokens, bias=bias) for n_tokens in l_embedding_tokens])
		self.names = l_embedding_names

	def indices(
		self,
		names
	):
		if isinstance( names[-1], int ):
			return names
		return [ self.names.index(name) for name in names ]

	def forward(
		self,
		xi: Tensor,
		levels: list[int] | None = None,
		names: list[str] | None = None,
		stack = False,
	) -> Tensor:
		dtype = xi[0].dtype
		device = xi[0].device

		if levels and isinstance( levels[-1], str ):
			names = levels
			levels = []

		# map names to levels
		if names and not levels:
			levels = [ None if name not in self.names else self.names.index(name) for name in names ]

		xi = [ x if l == None else self.proj[l]( x ) for x, l in zip(xi, levels) ]
		if not stack:
			return xi

		# pad if needed
		# to-do: validate that this causes ZERO issues
		# addendum: this does cause problems
		max_size = max([ x.shape[-1] for x in xi ])
		xi = [
			#x if l == 0 else
			x if x.shape[-1] == max_size else
			torch.cat( [x, torch.full( (x.shape[0], max_size - x.shape[-1]), -float("inf"), device=device, dtype=dtype) ], dim=-1 )
			for x, l in zip(xi, levels)
		]
		return torch.stack( xi )

def _dropout_codes( x, dropout_mask, dropout_token, swapped=False ):
	"""
	x = x.clone().detach().t()
	for l, t in enumerate( x ):
		x[l] = torch.where( dropout_mask, dropout_token, x[l] )
	return x.t()
	"""
	x = x.clone().detach()
	levels = x.shape[-1]
	for level in range( levels ):
		lhs = dropout_token if not swapped else x[..., level]
		rhs = x[..., level] if not swapped else dropout_token
		x[..., level] = torch.where( dropout_mask, lhs, rhs )
	return x

class Metrics(nn.Module):
	def __init__(
		self,
		l_embedding_tokens: int | list[int],
		top_k = 10,
		average="micro",
		multidim_average="global",
		ignore_index = -100
	):
		super().__init__()
		self.accuracy = nn.ModuleList([ MulticlassAccuracy(
			n_tokens,
			top_k=top_k,
			average=average,
			multidim_average=multidim_average,
			ignore_index=ignore_index,
		) for n_tokens in l_embedding_tokens ])
		self.precision = nn.ModuleList([ MulticlassPrecision(
			n_tokens,
			top_k=top_k,
			average=average,
			multidim_average=multidim_average,
			ignore_index=ignore_index,
		) for n_tokens in l_embedding_tokens ])

	def calc_accuracy( self, inputs, targets, classifier_levels ):
		return sum( [ self.accuracy[l]( input[:, :self.accuracy[l].num_classes], target ) for target, input, l in zip( targets, inputs, classifier_levels ) ] ) / len( inputs )
	
	def calc_precision( self, inputs, targets, classifier_levels ):
		return sum( [ self.precision[l]( input[:, :self.precision[l].num_classes], target ) for target, input, l in zip( targets, inputs, classifier_levels ) ] ) / len( inputs )

	def __call__(self, *args, **kwargs):
		return dict(
			acc=self.calc_accuracy(*args, **kwargs),
		)

class Base(nn.Module):
	def loss_factor(self, k):
		if self.config is None:
			return 1.0
		return self.config.loss_factor(k)

	def _prune(self, l: Tensor, stop = None):
		if stop is None:
			stop = self.stop_token

		indices = (l == stop).nonzero()

		if len(indices) == 0:
			return l

		return l[: indices.min().item()]

	def __init__(
		self,
		
		n_phn_tokens: int = 256,
		n_audio_tokens: int = 1024,
		n_text_tokens: int = 8575,

		d_model: int = 512,
		d_ffn: int = 4,
		n_heads: int = 8,
		n_layers: int = 12,
		p_dropout: float = 0.1,

		n_experts: int = 1,

		l_padding: int = 0,

		training = True,
		attention = None,
		config = None, 
	):
		super().__init__()
		self.training = training
		self.teaching = False
		self.config = config

		self.n_phn_tokens = n_phn_tokens
		self.n_audio_tokens = n_audio_tokens
		self.n_text_tokens = n_text_tokens

		self.d_model = d_model
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.n_experts = n_experts
		
		self.l_padding = l_padding

		self.ignore_index = -100

		self.n_resp_levels = self.config.resp_levels if self.config else n_resp_levels
		self.n_max_levels = self.config.max_levels if self.config else n_resp_levels
		self.capabilities = self.config.capabilities if self.config else ["ar", "nar", "len"]
		self.gradient_checkpointing = self.config.gradient_checkpointing if self.config is not None else True

		self.stop_token = self.n_audio_tokens
		self.mask_token = self.stop_token
		self.causal = True
		self.version = self.config.version if self.config is not None else 6
		self.causal_size = self.config.experimental.causal_size if self.config is not None else (1 if self.causal else 0)

		self.arch_type = self.config.arch_type if self.config is not None else "llama"

		# check if requested arch is unavailable
		if self.arch_type in ERROR_ARCHES:
			raise ERROR_ARCHES[self.arch_type]
		
		if not attention:
			attention = self.config.attention if self.config is not None else "auto"

		# crunge
		if self.config is not None and config.teacher:
			self.teaching = True
			self.training = False

		attention_backend = attention
		audio_embedding_sums = self.config.experimental.audio_embedding_sums if self.config is not None else False
		split_classifiers = self.config.experimental.split_classifiers if self.config is not None else False
		tie_classifier_to_embedding = self.config.experimental.tie_classifier_to_embedding if self.config is not None else False
		audio_embedding_mode = self.config.experimental.audio_embedding_mode if self.config is not None else ""
		unified_position_ids = self.config.experimental.unified_position_ids if self.config is not None else True
		#interleave = self.config.experimental.interleave if self.config is not None else False
		noncausal_masks = self.config.experimental.noncausal_masks if self.config is not None else False
		classifiers_bias = self.config.experimental.classifiers_bias if self.config is not None else False
		max_position_embeddings = self.config.experimental.max_position_embeddings if self.config is not None else (75 * 60 * 5)
		
		masking_ratio = self.config.experimental.masking_ratio if self.config is not None else False
		ignore_inputs_for_loss = self.config.experimental.ignore_inputs_for_loss if self.config is not None else False
		
		resp_parallel_training = self.config.experimental.resp_parallel_training if self.config is not None else True
		predict_causally = self.config.experimental.predict_causally if self.config is not None else False
		monolithic_audio_encoder = self.config.experimental.monolithic_audio_encoder if self.config is not None else False

		self.resp_parallel_training = resp_parallel_training
		self.predict_causally = predict_causally

		n_tasks = self.config.tasks if self.config is not None else 8
		n_langs = self.config.langs if self.config is not None else 2
		n_tones = self.config.tones if self.config is not None else 1
		
		n_resp_tokens = n_audio_tokens + ( 1 if self.causal_size > 0 else 0 )
		l_embedding_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1) + [n_resp_tokens]
		l_classifier_tokens = [n_resp_tokens] + [n_resp_tokens - 1] * (self.n_resp_levels - 1) + [n_resp_tokens - 1]
		l_embedding_names = ['AR:0:0'] + [f'NAR:{i}:{i+1}' for i in range( self.n_resp_levels - 1 )] + ['NAR:0:0']

		n_vocab = 17702 if not split_classifiers else n_resp_tokens + 1
		
		l_classifier_names = l_embedding_names

		# STT
		l_classifier_names += [ "stt" ]
		l_classifier_tokens += [ n_phn_tokens ]

		# LEN
		if "len" in self.capabilities:
			l_classifier_tokens += [ 11 ]
			l_classifier_names += ["len"]

		# TEXT => PHN / PHN => TEXT
		if self.version >= 6:
			l_classifier_tokens += [ n_text_tokens ]
			l_classifier_names = l_embedding_names + [ "text" ]

		self.n_vocab = n_vocab
		self.unified_position_ids = unified_position_ids
		self.inject_timestep_embedding = False # results in bad output
		self.masking_ratio = masking_ratio
		self.ignore_inputs_for_loss = ignore_inputs_for_loss
		self.noncausal_masks = noncausal_masks

		self.text_emb = Embedding(n_phn_tokens, d_model)
		self.raw_text_emb = None
		self.langs_emb = None
		self.tones_emb = None
		self.tasks_emb = None
		self.rvq_l_emb = None
		self.len_emb = None
		
		# it would be nicer for these to be a token or live inside an embedding
		self.sep = nn.Parameter(torch.randn(d_model))
		self.dropout_token = nn.Parameter(torch.randn(d_model))

		self.proms_emb = AudioEmbedding(
			[n_audio_tokens] * self.n_resp_levels, d_model,
			sums=audio_embedding_sums == "prom" or audio_embedding_sums == True,
		)
		self.resps_emb = AudioEmbedding(
			l_embedding_tokens, d_model,
			sums=audio_embedding_sums == "resp" or audio_embedding_sums == True,
			l_embedding_names=l_embedding_names,
		)

		self.langs_emb = Embedding(n_langs, d_model) if n_langs > 0 else None
		self.tasks_emb = Embedding(n_tasks, d_model) if n_tasks > 0 else None
		self.capabilities += ["lang"]
		# never actually got added... I kept forgetting to classify all my audio for speaker's tone
		self.tones_emb = Embedding(n_tones, d_model) if n_tones > 0 else None

		self.rvq_l_emb = Embedding(self.n_resp_levels, d_model)
		self.len_emb = Embedding(11, d_model)
		self.raw_text_emb = Embedding(self.n_text_tokens, d_model)

		if attention_backend == "auto":
			attention_backend = "sdpa"

		hf_attention = attention_backend
		HF_ATTENTIONS = ["eager", "sdpa", "flash_attention_2"]

		if attention_backend not in HF_ATTENTIONS:
			hf_attention = None
			if attention_backend not in AVAILABLE_ATTENTIONS:
				raise ValueError(f"Requesting attention `{attention_backend}` but is not available. Currently available: {AVAILABLE_ATTENTIONS}")

		# override any requested padding size
		if attention_backend == "flash_attn_v100":
			self.l_padding = 32
		elif attention_backend == "fused_attn":
			self.l_padding = 128


		self.model_config = LlamaConfig(
			vocab_size=0, # n_vocab,
			hidden_size=d_model,
			max_position_embeddings=max_position_embeddings,
			intermediate_size=d_model*d_ffn,
			num_hidden_layers=n_layers,
			num_attention_heads=n_heads,
			attention_dropout=p_dropout if training else 0.0,
			num_key_value_heads=n_heads,
			hidden_act="gelu",
			is_encoder_decoder=False,
			is_decoder=True,
			#gradient_checkpointing=self.gradient_checkpointing,
		)
		self.model_config.attn_mode = attention_backend
		self.model = LlamaModel(self.model_config)

		if not split_classifiers:
			self.classifier = nn.Linear(d_model, n_vocab, bias=classifiers_bias)
			self.classifiers = None
			self.metrics = None
		else:
			self.classifier = None
			self.classifiers = Classifiers( l_classifier_tokens, l_classifier_names, d_model, bias=classifiers_bias )
			self.metrics = Metrics( l_classifier_tokens )

	def _forward(
		self,
		inputs,
		mask = None,
		is_causal = None,
		position_ids = None,
		
		state = None,
		
		output_attentions = False,
		output_hidden_states = False,
	):
		x = inputs
		m = mask #.squeeze(-1).int()
		
		aux_loss = None
		attentions = None
		hidden_states = None

		# HF transformer derived model
		kwargs = dict(
			inputs_embeds=x,
			attention_mask=m,
			past_key_values=state,
			position_ids=position_ids,
			use_cache=False, # not self.training,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=True,
			is_causal=is_causal,
		)

		if self.n_experts > 1 and self.training:
			kwargs["output_router_logits"] = True

		output = self.model(**kwargs)
		x = output["last_hidden_state"]
		
		# to-do: figure out why KV caching doesn't work
		#if not self.training:
		if state is not None:
			state = output["past_key_values"]

		if output_attentions:
			attentions = output["attentions"]
		
		if output_hidden_states:
			hidden_states = output["hidden_states"]
		
		if self.n_experts > 1 and self.training:
			router_logits = output["router_logits"]
			aux_loss = self.model.config.router_aux_loss_coef * load_balancing_loss_func( router_logits, self.model.config.num_local_experts, self.model.config.num_experts_per_tok, m )

		# process it into a format that I like
		if output_hidden_states:
			# hidden_states is actually layers + 1, as hidden_states[0] == embedding...........
			hidden_states = [ state for state in hidden_states[1:] ]
			# apply normalization to these states (to-do: check if this matters)
			# but skip the last state, as it already is normalized
			hidden_states = [ x if i == self.n_layers - 1 else self.model.norm(output.hidden_states[i]) for i, state in enumerate( hidden_states ) ]

		return Logits(x, state, inputs, aux_loss, attentions, hidden_states)

	# takes a bunch of separate lists and parses them into an ordered array of tuples to guide input sequence creation
	def inputs(
		self,
		phns_list: list[Tensor] | None = None,
		text_list: list[Tensor] | None = None,

		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,

		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		task_list: list[str] | None = None,
		time_list: list[Tensor] | None = None,

		quant_levels: int | list[int] | Tensor | None = None
	):
		if phns_list and phns_list[0] is not None:
			device = phns_list[0].device
			batch_size = len(phns_list)
		elif text_list and text_list[0] is not None:
			device = text_list[0].device
			batch_size = len(text_list)
		elif proms_list and proms_list[0] is not None:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list and resps_list[0] is not None:
			device = resps_list[0].device
			batch_size = len(resps_list)

		inputs = [ [] for _ in range(batch_size) ]
		for i in range(batch_size):
			quant_level = quant_levels[i] if quant_levels is not None else 0
			task_type = task_list[i] if task_list is not None else "tts"
			timestep = time_list[i] if time_list is not None else None
			classifier_level = None

			# insert task type as a string
			inputs[i].append( ( "task", task_type ) )

			# to-do: maybe not split the below blocks up
			# might be beneficial in the event I need to use a difference sequence, such as STT tasks

			# Base-line TTS task
			# Sequence: <text><sep><rvq lvl><sep><prom><sep><resp>
			# prom /may/ include <task> tokens inside to help guide things, per SpeechX
			if task_type in get_task_symmap() and task_type not in special_tasks:
				# insert the text prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )
				elif text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert RVQ level guidance token if the model is versioned for it
				if self.rvq_l_emb is not None:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )

					classifier_level = "AR:0:0" if quant_level == 0 else f'NAR:{quant_level-1}:{quant_level}'
				# insert input audio prompt
				if proms_list is not None and proms_list[i] is not None:
					inputs[i].append( ( "prom", proms_list[i] ) )
				# insert tone token if we're trained for it
				if "tone" in self.capabilities and tone_list is not None and tone_list[i] is not None:
					inputs[i].append( ( "tone", tone_list[i] ) )
				# insert timestep token
				if timestep is not None:
					# force set to use this classifier level
					classifier_level = "NAR:0:0"
					# store timestep information
					if self.masking_ratio in ["random", "rand"]:
						# cosine scheduled timestep => masking ratio
						p = math.cos(timestep * math.pi * 0.5)
						# I don't think is is necessary as the timestep is encoded in the sequence by the number of masked tokens, probably.
						if self.inject_timestep_embedding:
							inputs[i].append( ("timestep", torch.tensor([timestep], device=device, dtype=self.time_emb.mlp[0].weight.dtype) ) )
					else:
						# a paper said to use a fixed masking ratio of 0.8 for training
						# ...but I want to make it user adjustable
						p = self.masking_ratio

					# store dropout mask (if training, as this gets used later to mask the input embeddings if provided)
					if self.training:
						dropout_mask = _dropout_mask( resps_list[i], p )
						inputs[i].append( ("dropout_mask", dropout_mask ) )
				# insert the current output response
				if resps_list is not None and resps_list[i] is not None:
					inputs[i].append( ( "resp", resps_list[i] ) )
				
				inputs[i].append( ("classifier_level", classifier_level) )
			# Audio length prediction task
			# Sequence: <text><sep><rvq lvl><prom><sep><len>
			elif task_type == "len":
				# throw an error so we don't silently train without this
				if self.len_emb is None:
					raise Exception(f"Requesting task `{task_type}` but corresponding embedding is not defined.")

				# insert the text prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )
				elif text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# technically will always be level 0 but for the sake of keeing the input formatting coherent...
				if self.rvq_l_emb is not None:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert input audio prompt
				if proms_list is not None and proms_list[i] is not None:
					inputs[i].append( ( "prom", proms_list[i] ) )
				# insert tone token if we're trained for it
				if "tone" in self.capabilities and tone_list is not None and tone_list[i] is not None:
					inputs[i].append( ( "tone", tone_list[i] ) )

				# insert output length tokens (if it exists)
				if len_list is not None and len_list[i] is not None:
					inputs[i].append( ( "len", len_list[i] ) )
				# "encode" length to tokens for 0-9 + stop
				elif resps_list is not None and resps_list[i] is not None:
					# yes this could be encoded better
					# 0,1,2,3,10(stop) => len = 123?
					inputs[i].append( ( "len", torch.tensor([ 0 ] + [ int(i) for i in str( resps_list[i].shape[0]) ] + [ 10 ], device=device, dtype=torch.int16) ) )
				
				inputs[i].append( ("classifier_level", "len") )
			# Speech-to-Text prediction task
			# Sequence: <resp><sep><rvq lvl><sep><text>
			elif task_type == "stt":
				# insert the input response
				if resps_list is not None and resps_list[i] is not None:
					inputs[i].append( ( "resp", resps_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				# insert RVQ level guidance token if the model is versioned for it
				if self.rvq_l_emb is not None:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert the output text prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )

				inputs[i].append( ("classifier_level", "phn") )
			# Text phonemizing task
			# Sequence: <text><sep><lang><sep><phonemes>
			elif task_type == "phn":
				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				if self.rvq_l_emb is not None:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert the text prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )

				inputs[i].append( ("classifier_level", "phn") )
			# Text de-phonemizing task
			# Sequence: <text><sep><lang><sep><phonemes>
			elif task_type == "un-phn":
				# insert the text prompt
				if phns_list is not None and phns_list[i] is not None:
					inputs[i].append( ( "phn", phns_list[i] ) )
				# insert lang token if we're trained for it
				if "lang" in self.capabilities and lang_list is not None and lang_list[i] is not None:
					inputs[i].append( ( "lang", lang_list[i] ) )
				if self.rvq_l_emb is not None:
					inputs[i].append( ( "quant_level", torch.tensor([ quant_level ], device=device, dtype=torch.int16) ) )
				# insert the text prompt
				if text_list is not None and text_list[i] is not None:
					inputs[i].append( ( "text", text_list[i] ) )
				
				inputs[i].append( ("classifier_level", "text") )
			else:
				raise Exception(f'Unrecognized task: {task_type}')
		return inputs

	def offset_inputs(
		self,
		inputs: list,
		direction: int = 1, # -1 to de-offset
	):
		offsets = _get_offsets()

		for batch_index, batch_input in enumerate(inputs):
			quant_level = None
			classifier_level = None
			# pre-iterate
			for name, input in batch_input:
				if name == "quant_level":
					quant_level = input
				elif name == "classifier_level":
					classifier_level = input

			for name, input in batch_input:
				if not isinstance( input, torch.Tensor ):
					continue

				k = name
				if name == "prom":
					k = f'prom|{quant_level}'
				elif name == "resp":
					k = f'resps|{classifier_level}'

				if k not in offsets:
					continue

				start, end = offsets[k]

				for i, t in enumerate( input ):
					input[i] += start * direction

		return inputs

	def inputs_to_embeddings(
		self,
		inputs: list, ##  alist of tenssor of shape=(8,)
		quant_levels: int | list[int] | Tensor | None = None
	):
		# handles tasks where the prompt has task tokens injected in the middle -- "#old models have the task tokens in the prom"
		def prompt_input_to_embedding( input, quant_level ):
			if isinstance(input, str):
				return self.tasks_emb( torch.tensor( [ get_task_symmap()[input] ], device=device, dtype=torch.int16) )

			# get RVQ level 0, or up to targetted RVQ level inference -- because "we need to sum up the embedding till quant_level"
			if self.version <= 4:
				return self.proms_emb(
					input if quant_level == 0 else input[:, :quant_level]
				)

			return self.proms_emb(
				input if input.dim() == 1 else input[:, : 1 if quant_level == 0 else quant_level],
				quant_level = 0 if quant_level == 0 else quant_level - 1, # input is one below the target quant level
				offset = 0,
			)

		# yuck
		token_dropout_rate = self.config.experimental.token_dropout_rate if self.config else 0.0
		token_dropout_rvq_levels = self.config.experimental.token_dropout_rvq_levels if self.config else None
		
		if self.dropout_token is None or not self.training:
			token_dropout_rate = 0.0

		if not token_dropout_rvq_levels:
			token_dropout_rvq_levels = [1, self.resp_levels]


		x_list = []
		for batch_index, batch_input in enumerate(inputs):
			batch = []
			quant_level = quant_levels[batch_index] if quant_levels is not None else 0
			
			task_type = "tts"
			input_prom = None
			classifier_level = None
			dropout_mask = None
			timestep = None
			
			# pre-iterate
			for name, input in batch_input:
				if name == "classifier_level": ## a batch will only process at the same level ?
					classifier_level = input
				elif name == "dropout_mask":
					dropout_mask = input
				elif name == "timestep":
					timestep = input

			for name, input in batch_input:
				# technically can provide a map for input_name => embedding, but some embedding requires additional processing
				embedding = None

				# is already an embedding		
				if name == "task":
					# noop
					# *maybe* inject a token for specifying task type
					task_type = input
					continue
				elif name == "phn":
					embedding = self.text_emb( input )

					device = embedding.device
				elif name == "text" and self.raw_text_emb is not None:
					embedding = self.raw_text_emb( input )

					device = embedding.device
				elif name == "quant_level" and self.rvq_l_emb is not None:
					embedding = self.rvq_l_emb( input )
				elif name == "lang" and self.langs_emb is not None:
					embedding = self.langs_emb( input )
				elif name == "prom":
					proms = [ input ] if isinstance(input, torch.Tensor) else input
					"""
					if proms is None:
						continue
					"""
					# to-do: probably insert separators if task requires it?
					embedding = torch.cat( [ prompt_input_to_embedding( input, quant_level ) for input in proms if input is not None ] )
				elif name == "tone" and self.tones_emb is not None:
					embedding = self.tones_emb( input )
				elif name == "resp":
					# if training NAR-len RVQ level 0
					if dropout_mask is not None:
						embedding = self.resps_emb(
							# if masked use masked token, else original token
							torch.where( dropout_mask, self.stop_token, input if input.dim() == 1 else input[:, quant_level] ),
							#quant_level = 0,
							name = classifier_level,
						)
					# NAR-len
					elif classifier_level == f"NAR:{quant_level}:{quant_level}":
						embedding = self.resps_emb(
							input if input.dim() == 1 else input[:, quant_level],
							#quant_level = 0,
							name = classifier_level,
						)
					# cheat-y way to handle performing STT across all levels
					elif task_type in summed_embeddings_task:
						# we do a manual sum because I trained it to use the AR embeddings + NAR embeddings for STT......
						embedding = sum([ self.resps_emb(
							input[:, :l+1],
							offset = 0 if l == 0 else 1, # or maybe set to 1
							quant_level = l,
							#name = 'AR:0:0' if l == 0 else f'NAR:{l-1}:{l}',
							sums = False
						) for l in range( input.shape[-1] - 1 ) ])
					else:
						# get RVQ level 0, or up to targetted RVQ level inference
						if self.version <= 4:
							embedding = self.resps_emb(
								input if quant_level == 0 else input[:, :quant_level],
								quant_level
							)
						else:
							input_quant_level = 0 if quant_level == 0 else quant_level - 1 # input is one below the target quant level
							embedding = self.resps_emb(
								input if input.dim() == 1 or quant_level == 0 else input[:, :quant_level],
								#offset = 0 if classifier_level.startswith("AR:") else 1,
								name = classifier_level,
								quant_level = input_quant_level,
							)

						# apply token dropout
						"""
						if token_dropout_rate > 0.0 and (token_dropout_rvq_levels[0] <= quant_level and quant_level <= token_dropout_rvq_levels[1]):
							steps = embedding.shape[0] - (1 if quant_level == 0 else 0) # do not mess with stop token
							for i in range( steps ):
								if random.random() > token_dropout_rate:
									continue
								embedding[i] = self.dropout_token
						"""
				elif name == "timestep" and self.time_emb is not None:
					embedding = self.time_emb( input )
				elif name == "len" and self.len_emb is not None:
					embedding = self.len_emb( input )
				else:
					# should probably raise an exception so things aren't processed silently
					continue

				batch.append(embedding)

			x_list.append( _join( batch, self.sep ) )

		return x_list

	# get an attribute from a given input list
	def get_input(
		self,
		inputs,
		name,
		at=None,
	):
		find_all = at is None
		res = [] if at is None else None
		
		for batch_index, batch_input in enumerate(inputs):
			if not find_all and batch_index != at:
				continue

			for n, input in batch_input:
				if n != name:
					continue
				if not find_all:
					return input
				res.append( input )
		
		return res

	# creates position ids from a given input list
	# if not unified_position_ids, then each input segment will have its own sequence
	def inputs_to_position_ids(
		self,
		inputs: list,
		mask: Tensor,
	):
		device = mask.device

		# shamelessly grabbed from modeling_llama.py
		ids = mask.long().cumsum(-1) - 1
		ids.masked_fill_( mask == 0, 1 )

		# there's a better way
		if not self.unified_position_ids:
			x_list = []

			def get_input_token_length( name, input, task ):
				# task token
				if isinstance(input, str):
					return 1

				# list of tokens
				if not isinstance(input, torch.Tensor):
					return sum( [ i.shape[0] for i in input if isinstance(i, torch.Tensor) ] )

				# ending input will not have a separator later
				return input.shape[0]

			for batch_index, batch_input in enumerate(inputs):
				# pre-iterate
				task = "tts"
				for name, input in batch_input:
					if name == "task":
						task = input

				batch = torch.cat( [
					torch.tensor([*range(get_input_token_length(name, input, task) + (1 if name != task_outputs.get(task, name) else 0))], device=device, dtype=torch.int32)
					for name, input in batch_input if name not in non_tokened_names
				] )

				delta = ids[batch_index].shape[0] - batch.shape[0]
				if delta > 0:
					batch = torch.cat( [ batch, torch.tensor([1] * delta, device=device, dtype=torch.int32) ] )

				x_list.append( batch )

			ids = torch.stack( x_list )

		return ids.to(device=device, dtype=torch.int32)

	def calc_loss(
		self,
		inputs: list,
		logits,
		
		quant_levels: list[int] | None = None,
		compute_hard_loss = True,
		compute_acc = True,
	):
		loss = {}
		stats = {}

		device = logits[0].device
		batch_size = len(logits)
		classifier_levels = self.get_input( inputs, "classifier_level" )

		# handles tasks where the prompt has task tokens injected in the middle
		def prompt_input_to_token( input, quant_level ):
			if isinstance(input, str):
				return torch.tensor( [ get_task_symmap()[input] ], device=device, dtype=torch.int16)

			# ignore prom, fill with mock tokens, because the prom embeddings don't directly map to tokens
			# prompt no gradient if sum!!!!!  even if transformer will have output for each input token
			if self.version < 4 or (self.version >= 5 and self.config and self.config.experimental.audio_embedding_sums):
				return torch.full_like(input[..., 0], self.ignore_index)

			return input if input.dim() == 1 else input[:, quant_level]

		def _calc_loss( logit, sequence, causal = True ):
			# filter tokens that exceed the vocab size
			sequence = torch.where( sequence >= logit.shape[-1], self.ignore_index, sequence )
			# drop if all tokens are ignored
			if all(sequence == self.ignore_index):
				return None, None

			# shift if causal
			if causal or self.predict_causally:
				l = self.causal_size
				logit = logit[..., :-l, :] # shift the target so that token n...
				sequence = sequence[..., l:] # ...predicts token n + 1

			nll = None
			metrics = None
			if compute_hard_loss:
				nll = F.cross_entropy( logit, sequence, ignore_index=self.ignore_index )

			if compute_acc:
				if self.metrics is not None and classifier_level in self.classifiers.names:
					metrics = self.metrics.calc_accuracy( [ logit ], [ sequence ], self.classifiers.indices([ classifier_level ]) )
				else:
					accuracy_metric = MulticlassAccuracy(
						logit.shape[-1],
						top_k = min(logit.shape[0], 10),
						average="micro",
						multidim_average="global",
						ignore_index = -100
					).to(logit.device)
					metrics = accuracy_metric( logit, sequence )

				metrics
			return nll, metrics
		
		for batch_index, batch in enumerate(inputs):
			quant_level = quant_levels[batch_index]
			target = []
			causal = True
			task_type = "tts"
			dropout_mask = None
			classifier_level = None
			output_len = 0

			for name, input in batch:
				if name == "task":
					task_type = input
				elif name == "dropout_mask":
					dropout_mask = input
				elif name == "classifier_level":
					classifier_level = input

			# autoregressive, causal
			if classifier_level.startswith("AR:"):
				causal = True
			# nonautoregressive, parallel
			elif classifier_level.startswith("NAR:"):
				causal = False

			it = 0
			for name, input in batch:
				token = None
				ignored = False

				# non-tokened tasks
				if name in non_tokened_names:
					continue
				# prom can either be a tensor itself or a list of tensors and strings
				if name == "prom":
					# expand to list if not a list
					proms = [ input ] if isinstance(input, torch.Tensor) else input
					# iterate over the list to inject their tokens
					token = torch.cat( [ prompt_input_to_token( input, quant_level ) for input in proms if input is not None ] )

					if logits[batch_index].dim() < 3 and token.dim() >= 2:
						token = token[..., 0]
				elif name == "resp":
					# mask found, apply it
					token = input if input.dim() == 1 else input[:, quant_level]
					
					# mask found, apply it
					if dropout_mask is not None:
						token = torch.where( dropout_mask, token, self.ignore_index )
				# not a special input, inject as-is
				else:
					token = input

				if not isinstance(token, torch.Tensor):
					continue

				if token.is_floating_point():
					ignored = True

				# grab range of our logits for later
				seq_len = token.shape[0]
				start, end = it, it+seq_len
				it += seq_len + 1 # +1 to incorporate the separator

				# deduce if a name for a task is an input or output
				if name != task_outputs.get(task_type, name):
					if self.ignore_inputs_for_loss:
						ignored = True
				else:
					output_len = seq_len

				if ignored:
					# pruned
					if self.config.loss_factors: ### will instead assign 0 weight so no need to set ignored????
						continue
					# fill with ignored out tensor
					token = torch.tensor( [ self.ignore_index ] * token.shape[0], device=device, dtype=torch.int16)

				# perform loss calculation on the individual piece
				if self.config.loss_factors:
					loss_factor = self.loss_factor(name)

					if loss_factor == 0.0:
						continue

					"""
					if name == "resp":
						name = f'{name}[{quant_level}]'
					"""

					nll, metrics = _calc_loss( logits[batch_index][start:end], token.long(), causal )

					if nll is not None:
						if f'{name}.nll' not in loss:
							loss[f'{name}.nll'] = []
						loss[f"{name}.nll"].append( nll * loss_factor )
					
					if metrics is not None:
						if f'{name}.acc' not in stats:
							stats[f'{name}.acc'] = []
						stats[f"{name}.acc"].append( metrics )
				# add to list
				else:
					target.append( token )
			
			# perofrm loss calculation on the entire sequence
			if not self.config.loss_factors:
				sequence = _join( target, torch.tensor(self.ignore_index, device=target[-1].device) )
				nll, metrics = _calc_loss( logits[batch_index], sequence, causal )
				
				if nll is not None:
					if 'nll' not in loss:
						loss['nll'] = []
					loss["nll"].append( nll )
				
				if metrics is not None:
					if 'acc' not in stats:
						stats['acc'] = []
					stats["acc"].append( metrics )

		# average
		loss = { name: sum( loss[name] ) / len( loss[name] ) for name in loss.keys() }
		stats = { name: sum( stats[name] ) / len( stats[name] ) for name in stats.keys() }

		return LossStats(loss, stats)

	def forward(
		self,
		inputs: list,

		quant_levels: list[int] | None = None,
		state: dict | list | None = None,
		
		output_attentions: bool = False,
		output_hidden_states: bool = False,
	):	
		# derive quant levels from inputs if not provided
		if quant_levels is None:
			quant_levels = [ x.item() for x in self.get_input( inputs, "quant_level" ) ]

		# inputs don't have quant levels added, pure AR
		if len(quant_levels) != len(inputs):  ## because NAR-training needs to sample from different quant_levels? weird. 
			quant_levels = [ 0 for _ in range(len(inputs)) ]

		x_list = self.inputs_to_embeddings( inputs, quant_levels )
		
		x, mask = list_to_tensor(x_list) ### this is important for batch processing in the pytorch models? always pad 0?

		training = self.training
		teaching = self.teaching
		device = x.device
		batch_size = len(x_list)

		# pad our input and mask, but retain the original length by doing it after
		if self.l_padding and x.shape[1] % self.l_padding != 0:
			# pad input
			shape = list(x.shape)
			shape[1] = self.l_padding - shape[1] % self.l_padding

			padding = torch.zeros(shape, dtype=x.dtype, device=x.device)
			x = torch.cat([x, padding], dim=1)

			# pad mask
			shape[2] = 1
			padding = torch.zeros(shape[:2], dtype=x.dtype, device=x.device)
			mask = torch.cat([mask, padding], dim=1)
		
		m = mask.unsqueeze(dim=-1)

		# needs to be done here as we still have our raw inputs
		position_ids = self.inputs_to_position_ids( inputs, mask=mask ) if not self.unified_position_ids else None ##for getting positional embedding later? ids can be unified/separated?
		classifier_levels = self.get_input( inputs, name="classifier_level" )
		causal_levels = [ "phn", "len", "phn" ] + [ f"AR:{_}:{_}" for _ in range( self.n_resp_levels) ]

		# right now limit to new versions because I need to retrain the model for noncausal masks...
		is_causal = [ l in causal_levels for l in classifier_levels ] if self.noncausal_masks else [ True for l in classifier_levels ]

		##hot fix for MDD
		# is_causal = [ l in casual_levels for l in classifier_levels ] if self.noncausal_masks else [ True for l in classifier_levels ]
		# assert len(is_causal) == 1
		# is_causal = is_causal[0]
  
		output = self._forward(
			inputs=x,
			mask=mask,
			state=state,
			is_causal=is_causal, ### is_causal will cause to generate the causal mask of input, it's 4D, BX1XTXT. corresponding to MHA B x H x Q x K 
			position_ids=position_ids,
			output_attentions = output_attentions,
		)

		logits = output.logits
		hidden_states = output.hidden_states
		
		# output projection layer
		# the very, very original implementation multiplied by the mask, but the mask only attends to padding, and the padding gets removed anyways
		if self.classifier is not None:
			logits = self.classifier(logits) # * m
		# to-do: piece-wise classification, now that there's a head for text
		# although again, one single monolithic head would be preferable instead......
		elif self.classifiers is not None:
			logits = self.classifiers(logits, levels = classifier_levels )

		# Remove padding
		logits = [ hi[..., :li, :] for hi, li in zip(logits, map(len, x_list)) ]
		
		if not training:
			loss = None
			stats = None

			self.loss = None
			self.stats = None

		# compute loss if the target is given
		else: ## each call only generate the loss for one level of codes
			loss, stats = self.calc_loss( inputs=inputs, logits=logits, quant_levels=quant_levels )

			# include any additional losses (for example: MoE router)
			if output.loss is not None:
				loss["aux_loss"] = output.loss

			self.loss = loss
			self.stats = stats
			
		# rewrap, because we're modifying the logits here
		return Logits(logits, output.state, inputs, loss, output.attentions, hidden_states)

	def sample(
		self,
		logits: list[Tensor], # logit scores
		prev_list: list[Tensor] | None = None, # previous tokens
		quant_levels: list[int] | None = None, # to-do: derive this from the prev_list
		**sampling_kwargs,
	):
		# yikes
		temperature = sampling_kwargs.get("temperature", 1.0)
		min_temperature = sampling_kwargs.get("min_temperature", -1.0)
		top_k = sampling_kwargs.get("top_k", -100)
		top_p = sampling_kwargs.get("top_p", 1.0)
		min_p = sampling_kwargs.get("min_p", 0.0)
		# repetition penalty parameters
		repetition_penalty = sampling_kwargs.get("repetition_penalty", 1.0)
		repetition_penalty_decay = sampling_kwargs.get("repetition_penalty_decay", 0.0)
		# length penalty parameters
		length_penalty = sampling_kwargs.get("length_penalty", 0.0)
		# beam sampling parameters
		beam_width = sampling_kwargs.get("beam_width", 0)
		# mirostat sampling parameters
		mirostat = sampling_kwargs.get("mirostat", None)
		# DRY sampling parameters
		dry_multiplier = sampling_kwargs.get("dry_multiplier", 0.0)
		dry_base = sampling_kwargs.get("dry_base", 1.75)
		dry_allowed_length = sampling_kwargs.get("dry_allowed_length", 2)
		#
		top_no = sampling_kwargs.get("top_no", 1.0)
		#
		attentions = sampling_kwargs.get("attentions", None)

		batch_size = len( logits )

		if min_temperature < 0:
			min_temperature = temperature

		# pick last RVQ level
		if prev_list is not None:
			prev_list = [ prevs if prevs.dim() == 1 else prevs[:, -1] for prevs in prev_list ]

		scores = None
		entropy = None
		#logits = [ logit.to(device="cpu", dtype=logit.dtype if logit.dtype != torch.float16 else torch.float32) for logit in logits ]
		#logits = [ logit.to(device="cpu") for logit in logits ]

		# (AR) entropix sampling
		# we do it before everything to retain logits for the entire sequence (even though it's still better to pass only the last token)
		if attentions is not None and quant_levels is None:
			# move to CPU for speedups
			seq_lens = [ logit.shape[0] for logit in logits ]
			attentions = torch.stack(attentions, dim=1).to(device="cpu") # ( batch, layer, heads, seq_len, seq_len )
			
			res = [ sample_entropix(
				logit[:seq_lens[batch], :], # ( seq_len, vocab )
				attentions[batch, :, :, :seq_lens[batch], :seq_lens[batch]], # (layer, heads, seq_len, seq_len )
				temperature,
				top_k,
				top_p,
				min_p,
			) for batch, logit in enumerate(logits) ]

			if res:
				return Sampled([ r[0] for r in res ], logits, scores, [ r[1] for r in res ])
		"""
		elif quant_levels is None:
			seq_lens = [ logit.shape[0] for logit in logits ]
			entropy = [ calculate_entropix_metrics(
				logit[:seq_lens[batch], :], # ( seq_len, vocab )
				#attentions[batch, :, :, :seq_lens[batch], :seq_lens[batch]], # (layer, heads, seq_len, seq_len )
			) for batch, logit in enumerate(logits) ]
		"""

		# (NAR) return the entire generated response
		# Parallel decoding relies on the last N tokens in the logits, because each token predicts the next RVQ layer in the same place (forgetfully obviously)	
		## transformer will generate the same number of tokens as input (prompts as well), only last L logtis are related to output???
		if quant_levels is not None: #  and "nar" in self.capabilities: # for when I get around to coping about dropping the NAR entirely
			seq_lens = map(len, prev_list)
			logits = [ logit[-l:] for logit, l in zip(logits, seq_lens) ]
		# (AR chunkwise) return the last chunkwise piece
		elif self.causal:
			seq_lens = [ logit.shape[0] - self.causal_size for logit in logits ]
			logits = [ logit[-self.causal_size:] for logit in logits ]

		# (NAR) disable stop token
		if quant_levels is not None:
			logits = [ ban_tokens(logit, tokens=[self.stop_token]) for logit, l in zip( logits, map(len, prev_list) ) ]
		# (AR-len) disable extraneous tokens
		"""
		if quant_levels is None and "len" in self.capabilities:
			logits = [ ban_tokens(logit, tokens=[*range(11, logit.shape[-1])]) for logit, l in zip( logits, map(len, prev_list) ) ]
		"""

		# perform repetition penalizing	
		if prev_list is not None and repetition_penalty != 1.0:
			logits = [ reptition_penalize(logit, previous=prevs, factor=repetition_penalty, decay=repetition_penalty_decay) for logit, prevs in zip( logits, prev_list ) ]

		# (AR) perform length penalizing
		if quant_levels is None and self.causal and prev_list is not None and length_penalty != 0.0:
			logits = [ length_penalize(logit, length=l + 1, factor=length_penalty, token=self.stop_token) for logit, l in zip( logits, map(len, prev_list) ) ]

		# perform min_p filtering of our logits
		if min_p > 0.0:
			logits = [ min_p_filtering(logit, min_p=min_p) for logit in logits ]

		# perform top_k/top_p filtering of our logits
		if top_k > 0 or top_p < 1.0:
			logits = [ top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p) for logit in logits ]	

		# trigger dynamic temperature sampling if the minimum temperature is not the same as the sampling temperature
		#	 epsilon float comparison because I don't trust Python
		if abs(temperature - min_temperature) >= 0.001: 
			logits = [ dynamic_temperature(logit, temperature=temperature, min_temperature=min_temperature) for logit in logits ]
		elif temperature > 0.0:
			logits = [ logit / temperature for logit in logits ]

		# do top-no logit processing
		if top_no > 0.0:
			logits = [ top_no_logits_processing(logit) for logit in logits ]

		# do DRY sampling
		if dry_multiplier > 0.0 and prev_list is not None:
			logits = [ dry_sampling(logit, previous=prevs, factor=dry_multiplier, base=dry_base, allowed_length=dry_allowed_length) for logit, prevs in zip( logits, prev_list ) ]

		# do mirostat sampling
		# currently incompatible with beam searching with the way the two are implemented, perhaps a night of brain bashing can make the two work
		if mirostat is not None:
			# mirostat sampling
			scores = [ mirostat_sample(logit, state=state) for logit, state in zip(logits, mirostat) ]
			res = [ state["token"] for state in scores ]
		# do beam search (naive implementation)
		# picks the top-k across all batches, and re-batches those resultant tokens
		# returns the logit scores as well to be P-concatted with the previous scores
		# to-do: not naively implement beam searching
		## strange can't understand it!
		elif beam_width > 1:
			candidates = top_k_logits_list( logits, beam_width )
			res = [ torch.tensor(token, dtype=torch.int16).unsqueeze(dim=-1) for batch, token in candidates ]
			scores = [ logits[batch].flatten()[token] for batch, token in candidates ]
		# basic sampling
		else:
			# argmax instead
			if temperature <= 0.0:
				res = [ logit.argmax(dim=-1) for logit in logits ]
			else:
				res = [ Categorical(logits=logit).sample() for logit in logits ]

			# calculate token probabilities
			scores = [
				F.softmax(logit, dim=-1).gather(1, tokens.unsqueeze(-1)).squeeze(-1)
				for logit, tokens in zip(logits, res)
			]

		return Sampled(res, logits, scores, entropy)

