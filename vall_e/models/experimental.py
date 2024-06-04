"""
This is an experiment to:
* entertain a thought to try and abide by HF's transformers API (to benefit from caching better)
* conform to a single embedding (instead of a bunch of them) by folding/unfolding inputs
* stop trying to make a mixed AR+NAR model work since it seems lobotomized if I keep trying to enforce both recurrent and parallel inferencing (despite a penalty cost)
	+ I will not cave and go with codebook patterns, not yet.
"""

from ..config import cfg

from ..data import fold_inputs, unfold_outputs

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint

import random
import math

from einops import rearrange
from tqdm import trange

AVAILABLE_ARCHES = []

try:
	from transformers import LlamaForCausalLM, LlamaConfig
	AVAILABLE_ARCHES.append("llama")
except Exception as e:
	print("Error importing `llama` arch:", e)
	pass

try:
	from .retnet_hf import RetNetConfig
	from ..ext.retnet_hf.modeling_retnet import RetNetForCausalLM

	AVAILABLE_ARCHES.append("retnet")
except Exception as e:
	print("Error importing `retnet` arch:", e)
	pass

try:
	from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig, MixerModel as MambaMixelModel, layer_norm_fn as MambaLayerNormFn, RMSNorm as MambaRMSNorm

	def MambaMixelModel_forward(self, input_ids, inference_params=None, **mixer_kwargs):
		hidden_states = self.embedding(input_ids)
		residual = None
		for layer in self.layers:
			if self.gradient_checkpointing and hidden_states.requires_grad:
				hidden_states, residual = checkpoint( layer, hidden_states, residual, inference_params=inference_params, use_reentrant=False )
			else:
				hidden_states, residual = layer( hidden_states, residual, inference_params=inference_params )
		if not self.fused_add_norm:
			residual = (hidden_states + residual) if residual is not None else hidden_states
			hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
		else:
			# Set prenorm=False here since we don't need the residual
			hidden_states = MambaLayerNormFn(
				hidden_states,
				self.norm_f.weight,
				self.norm_f.bias,
				eps=self.norm_f.eps,
				residual=residual,
				prenorm=False,
				residual_in_fp32=self.residual_in_fp32,
				is_rms_norm=isinstance(self.norm_f, MambaRMSNorm)
			)
		return hidden_states

	MambaMixelModel.forward = MambaMixelModel_forward

	AVAILABLE_ARCHES.append("mamba")
except Exception as e:
	print("Error importing `mamba` arch:", e)
	pass


SELECTED_ARCH = cfg.model.arch_type 
if SELECTED_ARCH not in AVAILABLE_ARCHES:
	raise ValueError(f"Requesting arch `{SELECTED_ARCH}` but not available")

if SELECTED_ARCH == "mamba":
	LlmArchClass = MambaLMHeadModel
elif SELECTED_ARCH == "llama":
	LlmArchClass = LlamaForCausalLM
elif SELECTED_ARCH == "retnet":
	LlmArchClass = RetNetForCausalLM
else:
	raise ValueError(f"Requesting arch `{SELECTED_ARCH}` but not available")

class Model(LlmArchClass):
	def __init__(
		self,
		d_model=1024,
		n_layers=12,
		n_heads=16,
		p_dropout=0.1,

		config = None,
	):
		self.hyper_config  = config
		
		hf_attention = config.attention if config is not None else None
		gradient_checkpointing = config.gradient_checkpointing if config is not None else True
		vocab_size = 256 + (1024 * cfg.model.max_levels) + (1024 * cfg.model.max_levels) + 1

		if SELECTED_ARCH == "llama":
			super().__init__(config=LlamaConfig(
				vocab_size=vocab_size,
				hidden_size=d_model,
				max_position_embeddings=cfg.dataset.frames_per_second * cfg.model.max_levels * 60, # max-length of 60 seconds
				intermediate_size=d_model*4,
				num_hidden_layers=n_layers,
				num_attention_heads=n_heads,
				attention_dropout=p_dropout,
				num_key_value_heads=n_heads,
				sliding_window=cfg.dataset.frames_per_second * cfg.model.max_levels * 12,
				hidden_act="gelu",
				is_encoder_decoder=False,
				is_decoder=True,
				attn_implementation=hf_attention,
			))
			
			if gradient_checkpointing:
				self.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
					use_reentrant=False
				))
		elif SELECTED_ARCH == "retnet":
			super().__init__(config=RetNetConfig(
				vocab_size=vocab_size,
				decoder_embed_dim=d_model,
				decoder_value_embed_dim =d_model * 2,
				decoder_retention_heads=n_heads,
				decoder_ffn_embed_dim=d_model * 4,
				decoder_layers=n_layers,
				dropout=p_dropout,
				checkpoint_activations=gradient_checkpointing,
				activation_fn="gelu",
				use_layernorm=False,
				use_biases=False,
				use_glu=True,

				#chunkwise_recurrent=self.causal and self.recurrent_chunk_size > 0,
				#recurrent_chunkwise_size=self.recurrent_chunk_size if self.causal else 0,
				#no_output_layer=True,
				#rotary_embedding_base=self.rotary_embedding_base, # 10000

				decoder_normalize_before=True,
			))
		elif SELECTED_ARCH == "mamba":
			super().__init__(config=MambaConfig(
				vocab_size=vocab_size,
				d_model=d_model,
				n_layer=n_layers*2,
				#ssm_cfg={"layer": "Mamba2"}, # will ALWAYS nan
			))

			self.backbone.gradient_checkpointing = gradient_checkpointing


	def forward(
		self,
		*args,
		**kwargs,
	):
		output = super().forward(*args, **kwargs)

		if SELECTED_ARCH in ["llama", "retnet"]:
			if output.loss is not None:
				self.loss = dict(
					nll = output.loss,
				)
		elif SELECTED_ARCH == "mamba":
			if "labels" in kwargs:
				labels = kwargs.pop("labels")
				logits = output.logits

				# Shift so that tokens < n predict n
				shift_logits = logits[..., :-1, :].contiguous()
				shift_labels = labels[..., 1:].contiguous()
				# Flatten the tokens
				loss_fct = CrossEntropyLoss()
				shift_logits = shift_logits.view(-1, shift_logits.size(-1))
				shift_labels = shift_labels.view(-1)
				# Enable model parallelism
				shift_labels = shift_labels.to(shift_logits.device)
				loss = loss_fct(shift_logits, shift_labels)

				self.loss = dict(
					nll = loss,
				)

		return output

def example_usage():
	cfg.trainer.backend = "local"
	cfg.hyperparameters.gradient_accumulation_steps = 1
	if cfg.audio_backend == "dac":
		cfg.sample_rate = 44_000

	from functools import partial
	from einops import repeat
	from tqdm import tqdm

	from ..emb.qnt import decode_to_file, unload_model
	from ..engines import Engine
	from ..utils import wrapper as ml
	
	import numpy as np
	import re

	device = "cuda"

	def tokenize(content):
		return torch.tensor( cfg.tokenizer.encode(content) )

	def _load_quants(path) -> Tensor:
		qnt = np.load(path, allow_pickle=True)[()]
		return torch.from_numpy(qnt["codes"].astype(np.int16))[0, :cfg.model.max_levels, :].t().to(torch.int16)

	qnt = _load_quants(f"./data/qnt.{'dac' if cfg.audio_backend == 'dac' else 'enc'}")


	text_list = [
		tokenize("ˈaɪ wɪl nˌɑːt ˈæsk ɐ sˈɛkənd tˈaɪm").to(device),
		#tokenize("ˈaɪ wɪl nˌɑːt ˈæsk ɐ sˈɛkənd tˈaɪm").to(device),
	]
	proms_list = [
		qnt[:cfg.dataset.frames_per_second, :].to(device),
		#qnt[:cfg.dataset.frames_per_second, :].to(device),
	]
	resps_list = [
		qnt[:, :].to(device),
		#qnt[cfg.dataset.frames_per_second:, :].to(device),
	]

	text_list = text_list[:1]
	proms_list = proms_list[:1]
	resps_list = resps_list[:1]

	input_ids, attention_mask = fold_inputs(text_list, proms_list, resps_list)
	target_ids, target_attention_mask = fold_inputs(text_list, proms_list, resps_list, ignore_index=-100)
	prefix_input_ids, prefix_attention_mask = fold_inputs(text_list, proms_list)

	kwargs = {}
	model = Model(**kwargs).to(device)
	steps = 50

	optimizer = cfg.hyperparameters.optimizer.lower() if cfg.cfg_path is not None else "prodigy"
	scheduler = cfg.hyperparameters.scheduler.lower() if cfg.cfg_path is not None else ""
	learning_rate = cfg.hyperparameters.learning_rate if cfg.cfg_path is not None else None

	if cfg.optimizations.dadaptation:
		# do not combine the two
		if scheduler == "schedulefree":
			scheduler = ""

		learning_rate = 1.0
	
	if optimizer == "prodigy":
		if learning_rate is None:
			learning_rate = 1.0

		optimizer = ml.Prodigy
	elif optimizer == "adagrad":
		if learning_rate is None:
			learning_rate = 1.0e-2

		optimizer = ml.Adagrad
	elif optimizer == "adamw":
		if learning_rate is None:
			learning_rate = 1.0e-4

		optimizer = ml.AdamW
	elif optimizer == "sdg":
		if learning_rate is None:
			learning_rate = 1.0e-4

		optimizer = ml.SGD
	else:
		raise ValueError(f"Unrecognized optimizer: {optimizer}")

	print("Optimizer:", optimizer, "\tLearning rate:", learning_rate)

	optimizer = optimizer(model.parameters(), lr=learning_rate)

	if scheduler == "schedulefree":
		if isinstance(optimizer, ml.AdamW):
			scheduler = ml.schedulefree.AdamWScheduleFree
		elif isinstance(optimizer, ml.SGD):
			scheduler = ml.schedulefree.SGDScheduleFree
		else:
			scheduler = None

		if scheduler is not None:
			print("Scheduler:", scheduler)
			optimizer = scheduler( model.parameters(), lr = learning_rate )

	if cfg.optimizations.replace and cfg.optimizations.linear:
		model = ml.replace_linear( model )
		
	if cfg.optimizations.replace and cfg.optimizations.embedding:
		model = ml.replace_embedding( model )
	
	engine = Engine(model=model, optimizer=optimizer)

	torch.save( {
		'module': model.state_dict()
	}, f"./data/{SELECTED_ARCH}.pth" )

	print(f"{LlmArchClass} parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

	@torch.inference_mode()
	def sample( name, steps=cfg.model.max_levels*cfg.dataset.frames_per_second*60 ):
		engine.eval()
		if SELECTED_ARCH == "mamba":
			output = model.generate(input_ids=prefix_input_ids, cg=True, max_length=steps, eos_token_id=3)
		else:
			output = model.generate(input_ids=prefix_input_ids, attention_mask=prefix_attention_mask, max_length=steps, eos_token_id=3, do_sample=False)

		unfolded = unfold_outputs( output )
		for i, batch in enumerate(unfolded["resp_list"]):
			_ = decode_to_file(batch.to(device=device), f"data/{SELECTED_ARCH}.{cfg.audio_backend}.{i}.{name}.wav", device=device)

		unload_model()

	def train():
		engine.train()
		t = trange(steps)
		for i in t:
			stats = {"step": i}
			if SELECTED_ARCH == "mamba":
				stats |= engine.traverse(input_ids=input_ids, labels=target_ids)
			else:
				stats |= engine.traverse(input_ids=input_ids, labels=target_ids, attention_mask=attention_mask)
			stats |= {"grad_norm": engine.get_global_grad_norm()}

			tqdm.write(f"{stats}")

		torch.save( {
			'module': model.state_dict()
		}, f"./data/{SELECTED_ARCH}.pth" )

	#sample("init", 5)
	train()
	sample("final")

if __name__ == "__main__":
	example_usage()
