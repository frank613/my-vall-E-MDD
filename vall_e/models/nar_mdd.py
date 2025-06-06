"""
# an AR + NAR model that handles:
* inferencing the primary RVQ level in an autoregressive manner (AR)
* inferencing the remaining RVQ levels in parallel (NAR)

This model can fully handle being trained as a unified model (AR + NAR) or separate models (AR | NAR).
It's recommended to train as a unified model, then "distill" knowledge of each tasks separately, just in case.
"""
from .base_mdd import Base, list_to_tensor, Categorical
from ..config import cfg

import torch
from torch.nn.utils.rnn import pad_sequence

import random
import math
import time
import sys
from einops import rearrange
from torch import Tensor
from tqdm import trange, tqdm

import logging
import pdb

_logger = logging.getLogger(__name__)

#from ..emb.qnt import trim, encode_as_embedding, get_silence
from ..emb.qnt import trim, get_silence
from ..utils import get_devices, setup_logging, timer, clamp, convert_kwargs

from .lora import enable_lora
from ..samplers import cfg_logits, cfg_logits_modified

text_task = [ "stt", "phn", "un-phn" ]
##define it here for the ease 
def mdd_mask( pid_seq, index, length, mask_ratio, device):
    mask = torch.full((length,), False, dtype=torch.bool, device=device )
    pid, l, r = pid_seq[index]
    if mask_ratio < 1:
        sys.exit("mask_ratio must greater than 1") 
    extend = math.floor((r-l) * (mask_ratio - 1) / 2)
    l = l - extend if l - extend >= 0 else 0
    r = r + extend if r + extend <= length else length
    mask[l:r] = True ## 1 is masked! same as above, different from below, because later will we use "where" operation 
    return mask

## compute gop for mdd-nar-v2
def compute_gop(logit, resps, device=None):
	#logit: (T+?) x V , resps: T
	#return the T  gop_score for each frame
	if device is None:
		device = resps.device
	len = resps.shape[0]
	assert logit.shape[0] >= len
	logit_temp = logit[-len:, :].softmax(dim=1)
	one_hot = torch.zeros_like(logit_temp, device=device).scatter_(1,resps[:,None].type(torch.int64),1)
	return (logit_temp*one_hot).sum(dim=1)

##define it here for the ease
def compute_cfg_posterior(logits, pid_seq, resps_list):
	assert len(logits) == len(pid_seq) and len(resps_list) == len(pid_seq)
	device = logits[0].device
	avg_post_list = []
	pooled_list = []
	for batch_index, logit in enumerate(logits):
		seq_len = resps_list[batch_index].shape[0]
		logit = logit[-seq_len:]
		mask = mdd_mask(pid_seq, batch_index, seq_len, 1, device) 
		assert mask.shape[0] == seq_len
		index1 = torch.nonzero(mask==True, as_tuple=True)[0].tolist()
		index2 = resps_list[batch_index][mask].squeeze().tolist()
		avg_posterior = (logit - logit.logsumexp(dim=-1, keepdim=True))[index1, index2].mean()
		avg_post_list.append(avg_posterior.item())

		pooled_value = logit[index1[0]:index1[0]+len(index1), :].softmax(dim=-1).mean(dim=0)[index2].mean().log().item()
		pooled_list.append(pooled_value)
	return avg_post_list, pooled_list

class AR_NAR_MDD(Base):
	# yikes
	def forward_super(self, *args, **kwargs):
		return super().forward(*args, **kwargs)

	def forward_mdd_nar(
		self,

		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,
		pid_seq: list[list] | None = None,
		is_masking_nar_level_0: bool | None = None,
		disable_tqdm=False,
		use_lora=None,
		total_levels=None,
		cfg_strength_lv0=None,
		mask_ratio_lv0=None,
		diff_symbol=None,
		**sampling_kwargs,
	):
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.75)
		# deduce batch_size
		if text_list:
			device = text_list[0].device
			batch_size = len(text_list)
		elif raw_text_list:
			device = raw_text_list[0].device
			batch_size = len(raw_text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list)
   
		assert all([True if task == "tts" else False for task in task_list])
		assert self.config.experimental.token_dropout_error == 0 and self.config.experimental.token_dropout_rate == 0
  
		if total_levels is None:
			sys.exit("must specify the number of code levels for computing GOP")
	
		avg_post_list = []
		pooled_list = []
		iterator = trange(total_levels, desc="NAR")
		for n in iterator:
			level = n
			quant_levels = [ level for i in range(batch_size)]
			resps_list_in = [ r[..., :level+1] for r in resps_list] ##we constrain it here although in base model, it only sums up untill quant_level as resp embeddings
			if level == 0 and is_masking_nar_level_0:  ## NAR-mask for level 0
				### timesteps embedding as input are deprecated in versions >= 5
				### here for MDD we simply remove timesteps as input
				inputs = self.inputs(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list_in,
					lang_list=lang_list,
					tone_list=tone_list,
					task_list=task_list,
					raw_text_list=raw_text_list,
					quant_levels=quant_levels,
					pid_seq = pid_seq,
					is_nar_level_0 = True, ##False if level!=0 or AR for level 0
					masking_resp_mdd = True,
					compute_mdd = True,
					mask_ratio_lv0 = mask_ratio_lv0,
				)

				output = super().forward(
					inputs=inputs
				)
				avg_posteriors, pooled_posteriors = output[3]
		
				
				##cfg=null prompt, does not need masking
				if cfg_strength_lv0 is not None and cfg_strength_lv0 > 0:
					logits = output.logits
					null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
					null_prom = [ None for _ in range(batch_size) ]
					null_inputs = self.inputs(
						text_list=null_text,
						proms_list=null_prom,
						resps_list=resps_list_in,
						lang_list=lang_list,
						tone_list=tone_list,
						task_list=task_list,
						raw_text_list=raw_text_list,
						quant_levels=quant_levels,
						pid_seq = pid_seq,
						is_nar_level_0 = True, ##False if level!=0 or AR for level 0
						masking_resp_mdd = True,
						compute_mdd = True,
					)
					null_output = super().forward(
						inputs=null_inputs
					)
					logits_null = null_output.logits
					###number of phonemes
					assert len(logits) == len(logits_null)
					len_list = [ resps.shape[0] for resps in resps_list_in ]
					logits = cfg_logits( logits=logits, null=logits_null, strength=cfg_strength_lv0, rescale=cfg_rescale, lens=[ l for l in len_list ] )
					##Do it here for convieniance (originially need to be done in base class)
					avg_posteriors, pooled_posteriors = compute_cfg_posterior(logits, pid_seq, resps_list_in)

				if diff_symbol is not None:
					##currently support cfg0 for diff only
					#assert cfg_strength_lv0 is None
					if diff_symbol == "null":
						diff_text = [torch.tensor([1, 2], device=device)] * len(text_list)
					else:
						diff_text = []
						for i,phns in enumerate(text_list):
							diff_phns = phns.clone()
							diff_phns[:] = diff_symbol
							diff_phns[0] = 1
							diff_phns[-1] = 2
							diff_text.append(diff_phns)
      
					inputs = self.inputs(
						text_list=diff_text,
						proms_list=proms_list,
						resps_list=resps_list_in,
						lang_list=lang_list,
						tone_list=tone_list,
						task_list=task_list,
						raw_text_list=raw_text_list,
						quant_levels=quant_levels,
						pid_seq = pid_seq,
						is_nar_level_0 = True, ##False if level!=0 or AR for level 0
						masking_resp_mdd = True,
						compute_mdd = True,
						mask_ratio_lv0 = mask_ratio_lv0,
					)
					output = super().forward(
						inputs=inputs
					)
					avg_posteriors_diff, pooled_posteriors_diff = output[3]
					#avg_posteriors = [avg_posteriors[i] - avg_posteriors_diff[i] for i in range(len(avg_posteriors_diff)) ]
     
					if cfg_strength_lv0 is not None and cfg_strength_lv0 > 0:
						logits = output.logits	
						###number of phonemes
						assert len(logits) == len(logits_null)
						len_list = [ resps.shape[0] for resps in resps_list_in ]
						logits = cfg_logits( logits=logits, null=logits_null, strength=cfg_strength_lv0, rescale=cfg_rescale, lens=[ l for l in len_list ] )
						##Do it here for convieniance (originially need to be done in base class)
						avg_posteriors_diff, pooled_posteriors_diff = compute_cfg_posterior(logits, pid_seq, resps_list_in)
      
					avg_posteriors = [avg_posteriors[i] - avg_posteriors_diff[i] for i in range(len(avg_posteriors_diff)) ]	
					pooled_posteriors = [pooled_posteriors[i] - pooled_posteriors_diff[i] for i in range(len(pooled_posteriors_diff)) ]
	
					
	
			elif not is_masking_nar_level_0:
				sys.exit("only support is_masking_nar_level_0 = True")
			else:  ## other NAR levels
				### timesteps embedding as input are deprecated in versions >= 5
				### here for MDD we simply remove timesteps as input
				inputs = self.inputs(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list_in,
					lang_list=lang_list,
					tone_list=tone_list,
					task_list=task_list,
					raw_text_list=raw_text_list,
					quant_levels=quant_levels,
					pid_seq = pid_seq,
					is_nar_level_0 = False,
					masking_resp_mdd = False, ## wether to mask the resps segment derived from FA
					compute_mdd = True,
				)
				Logits = super().forward(
					inputs=inputs
				)
				avg_posteriors, pooled_posteriors = Logits[3]
    
				if diff_symbol is not None:
					if diff_symbol == "null":
						diff_text = [torch.tensor([1, 2], device=device)] * len(text_list)
					else:
						diff_text = []
						for i,phns in enumerate(text_list):
							diff_phns = phns.clone()
							diff_phns[:] = diff_symbol
							diff_phns[0] = 1
							diff_phns[-1] = 2
							diff_text.append(diff_phns)
					## resps_list_in has the resps up to quant_level, it will reduce one level in "inputs_to_embeddings"
					# reduced level is used for computing avg_posterior
					inputs = self.inputs(
						text_list=diff_text,
						proms_list=proms_list,
						resps_list=resps_list_in,
						lang_list=lang_list,
						tone_list=tone_list,
						task_list=task_list,
						raw_text_list=raw_text_list,
						quant_levels=quant_levels,
						pid_seq = pid_seq,
						is_nar_level_0 = False,
						masking_resp_mdd = False, ## wether to mask the resps segment derived from FA
						compute_mdd = True,
					)
					output = super().forward(
						inputs=inputs
					)
					avg_posteriors_diff, pooled_posteriors_diff = output[3]
					avg_posteriors = [avg_posteriors[i] - avg_posteriors_diff[i] for i in range(len(avg_posteriors_diff)) ]
					pooled_posteriors = [pooled_posteriors[i] - pooled_posteriors_diff[i] for i in range(len(pooled_posteriors_diff)) ]
     
			assert len(avg_posteriors) == len(pid_seq)
			avg_post_list.append(avg_posteriors)
			pooled_list.append(pooled_posteriors)
   
		return avg_post_list,pooled_list
	
	def forward_mdd_nar_v2(
		self,
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		disable_tqdm=False,
		use_lora=None,
  
		total_levels=None,
		cfg_strength_gop=None,
		diff_symbol=None,
		phoneme_mask_list=None,
		n_step_level_0=None,
	):
		cfg_rescale = 0.75
		cfg_strength = cfg_strength_gop
		temperature = 0
		# deduce batch_size
		if text_list:
			device = text_list[0].device
			batch_size = len(text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list)
   
		assert self.config.experimental.token_dropout_error == 0 and self.config.experimental.token_dropout_rate == 0
		assert total_levels == resps_list[0].shape[-1]
		min_length = 1
		max_length = 5000

		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]
		len_list = [ clamp(resps.shape[0], min_length, max_length) for resps in resps_list]
		gop_list = []
		gop_diff_list = []
		##iterations
		iterator = trange(total_levels, desc="NAR")
		for n in iterator:
			level = n
			quant_levels = [ level for i in range(batch_size)]
			##we constrain it here although in base model, it only sums up untill quant_level as resp embeddings, 
			# he last level is used for computing GOP
			resps_list_in = [ r[..., :level+1] for r in resps_list] 
			if level == 0 :  ## for level-0 NAR, masked. Similar to v2, but here the input is only one layer: list of  T x 1 tensor
				largest_score = 1.0
				smallest_score = 0.0 # -float("inf")
				start_noise_p = [phoneme_mask.sum()/len for phoneme_mask, len in zip(phoneme_mask_list, len_list) ]
				start_noise = [ math.acos(p)/ (math.pi * 0.5) for p in start_noise_p ]
				end_noise = [1] * batch_size
				max_steps = n_step_level_0 + 1 
				linspace_list_orig = [ torch.linspace(start, end, max_steps)[1:].tolist() for start, end in zip(start_noise, end_noise)]
				# P x step -> step X P list
				linspace_list = list(map(list, zip(*linspace_list_orig)))
				##initial masks
				masked_resps_list = [torch.where( phoneme_mask, self.stop_token, resps.squeeze()) for resps, phoneme_mask in zip(resps_list_in, phoneme_mask_list)]
				masked_resps_list_diff = [torch.where( phoneme_mask, self.stop_token, resps.squeeze()) for resps, phoneme_mask in zip(resps_list_in, phoneme_mask_list)]
				is_masked = [ resps == self.stop_token for resps in masked_resps_list ]
				is_masked_diff = [ resps == self.stop_token for resps in masked_resps_list_diff ]
				time_list = start_noise
				out_probs = [ torch.zeros_like(resp.squeeze(), device=device) for resp in resps_list_in]
				out_probs_diff = [ torch.zeros_like(resp.squeeze(), device=device) for resp in resps_list_in]
		
				iterator_nar = trange(max_steps-1, desc="NAR-MDD")
				for step in iterator_nar:
					new_time_list = linspace_list[step]
					mask_p_list = [ math.cos( timestep * math.pi * 0.5 ) for timestep in new_time_list]
					# full condition section
					inputs = super().inputs(
						text_list=text_list,
						proms_list=proms_list,
						resps_list=masked_resps_list,
						lang_list=lang_list,
						tone_list=tone_list,
						time_list=time_list,
						quant_levels=quant_levels,
					)
					output = super().forward(
						inputs=inputs,
					) 
					#pdb.set_trace()
					#logits = output.logits
					if cfg_strength > 0:
						null_inputs = super().inputs(
							text_list=null_text,
							proms_list=null_prom,
							resps_list=masked_resps_list,
							lang_list=lang_list,
							tone_list=tone_list,
							time_list=time_list,
							quant_levels=quant_levels,
						)
						null_output = super().forward(
							inputs=null_inputs,
						)

						logits = cfg_logits_modified( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )
						
					else:
						logits = output.logits
					# compute GOP scores, only updating tokens that were masked off, and force keeping unmasked tokens
					gop_scores = [ torch.where( is_masked, compute_gop(logit, resp.squeeze(), device), largest_score ) for logit, is_masked, resp in zip( logits, is_masked, resps_list_in ) ]
					# prepare for the next step
					if new_time_list[0] != 1: 
						# pick the worst scoring tokens to mask off, seq_len - (step+1) because we start from step 1 actually
						masked_indices = [ score.topk( clamp( int( mask_p * seq_len ), 1, seq_len - step - 1), dim=0, largest=False ).indices for score, seq_len, mask_p in zip(gop_scores, len_list, mask_p_list) ]
						masked_resps_list = [ resp.squeeze().scatter(0, indices, self.mask_token) for resp, indices in zip( resps_list_in, masked_indices ) ]
						new_masked = [ resps == self.mask_token for resps in masked_resps_list ]
						fix_masked = [ old.logical_xor(new) for new, old in zip(new_masked, is_masked)] 
						out_probs = [ out_prob + torch.where(fix_mask, gop_score ,0) if out_prob[fix_mask].sum()==0 else sys.exit("refill already fixed tokens") for fix_mask, gop_score, out_prob in zip(fix_masked, gop_scores, out_probs)]
						is_masked = new_masked
				
					else:
					## last step done, we don't mask anymore。 collect all the gops using is_masked from previous
						fix_masked = is_masked
						out_probs = [ out_prob + torch.where(fix_mask, gop_score ,0) if out_prob[fix_mask].sum()==0 else sys.exit("refill already fixed tokens") for fix_mask, gop_score, out_prob in zip(fix_masked, gop_scores, out_probs)]

					## diff section
					assert diff_symbol is not None
					if diff_symbol == "null":
						diff_phns_list = [torch.tensor([1, 2], device=device)] * batch_size
					else:
						diff_phns = torch.ones_like(text_list[0], device=device) * diff_symbol 
						diff_phns[0] = 1
						diff_phns[-1] = 2
						diff_phns_list = [diff_phns] * batch_size
						
					inputs_diff = super().inputs(
						text_list=diff_phns_list,
						proms_list=proms_list,
						resps_list=masked_resps_list_diff,
						lang_list=lang_list,
						tone_list=tone_list,
						time_list=time_list,
						quant_levels=quant_levels,
					)
					output_diff = super().forward(
						inputs=inputs_diff,
					)

					#logits_diff = output_diff.logits
					if cfg_strength > 0:
						null_inputs_diff = super().inputs(
							text_list=null_text,
							proms_list=null_prom,
							resps_list=masked_resps_list_diff,
							lang_list=lang_list,
							tone_list=tone_list,
							time_list=time_list,
							quant_levels=quant_levels,
						)
						null_output_diff = super().forward(
							inputs=null_inputs_diff,
						)

						logits_diff = cfg_logits_modified( logits=output_diff.logits, null=null_output_diff.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )
					else:
						logits_diff = output_diff.logits
			
					# compute GOP scores, only updating tokens that were masked off, and force keeping unmasked tokens
					gop_scores_diff = [ torch.where( is_masked, compute_gop(logit, resp.squeeze(), device), largest_score ) for logit, is_masked, resp in zip( logits_diff, is_masked_diff, resps_list_in ) ]
					#gop_scores_diff_nocfg = [ torch.where( is_masked, self.compute_gop(logit, resp, device), largest_score ) for logit, is_masked, resp in zip( output_diff.logits, is_masked_diff, resps_list ) ]
					# prepare for the next step
					if new_time_list[0] != 1: 
						# pick the worst scoring tokens to mask off, seq_len - (step+1) because we start from step 1 actually
						masked_indices_diff = [ score.topk( clamp( int( mask_p * seq_len ), 1, seq_len - step - 1), dim=0, largest=False ).indices for score, seq_len, mask_p in zip(gop_scores_diff, len_list, mask_p_list) ]
						masked_resps_list_diff = [ resp.squeeze().scatter(0, indices, self.mask_token) for resp, indices in zip( resps_list_in, masked_indices_diff ) ]
						new_masked_diff = [ resps == self.mask_token for resps in masked_resps_list_diff ]
						fix_masked_diff = [ old.logical_xor(new) for new, old in zip(new_masked_diff, is_masked_diff)] 
						out_probs_diff = [ out_prob + torch.where(fix_mask, gop_score ,0) if out_prob[fix_mask].sum()==0 else sys.exit("refill already fixed tokens") for fix_mask, gop_score, out_prob in zip(fix_masked_diff, gop_scores_diff, out_probs_diff)]
						is_masked_diff = new_masked_diff
					else:
					## last step done, we don't mask anymore。 collect all the gops
						fix_masked_diff = is_masked_diff
						out_probs_diff = [ out_prob + torch.where(fix_mask, gop_score ,0) if out_prob[fix_mask].sum()==0 else sys.exit("refill already fixed tokens") for fix_mask, gop_score, out_prob in zip(fix_masked_diff, gop_scores_diff, out_probs_diff)]
					
					### update timelist
					time_list = new_time_list			
  
				##collect GOP FOR LEVEL 0
				gop_list.append(out_probs)
				gop_diff_list.append(out_probs_diff)
    
			## other levels
			else:  ## other NAR levels
				### timesteps embedding as input are deprecated in versions >= 5, here we must have it because of using base model not base_mdd, because it signifies the using of NAR
				inputs = self.inputs(
					text_list=text_list[:1],
					proms_list=proms_list[:1],
					resps_list=resps_list_in[:1],
					lang_list=lang_list[:1],
					quant_levels=quant_levels[:1],
					time_list = [1]
				)
				output = super().forward(
					inputs=inputs
				)
	
				## we need to implement it here in this version, no!! I think return all the frame-wise probs is better, as done in v2
				assert len(output.logits) == 1
				out_probs = compute_gop((output.logits)[0], resps_list_in[0][:, -1], device)
    
				if diff_symbol is not None:
					if diff_symbol == "null":
						diff_text = [torch.tensor([1, 2], device=device)] * len(text_list)
					else:
						diff_text = []
						for i,phns in enumerate(text_list):
							diff_phns = phns.clone()
							diff_phns[:] = diff_symbol
							diff_phns[0] = 1
							diff_phns[-1] = 2
							diff_text.append(diff_phns)
      
					inputs = self.inputs(
						text_list=diff_text[:1],
						proms_list=proms_list[:1],
						resps_list=resps_list_in[:1],
						lang_list=lang_list[:1],
						quant_levels=quant_levels[:1],
					)
					output_diff = super().forward(
						inputs=inputs
					)
					assert len(output_diff.logits) == 1
					out_probs_diff = compute_gop((output_diff.logits)[0], resps_list_in[0][:, -1], device)
     
				## * batch_size because we want to align it with level 0 where each phoneme has different logits because of masking
				gop_list.append([out_probs]*batch_size)
				gop_diff_list.append([out_probs_diff]*batch_size)
   
		return gop_list,gop_diff_list
  
	def forward_mdd_nar_nomask(
		self,
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		disable_tqdm=False,
		use_lora=None,
  
		total_levels=None,
		cfg_strength_gop=None,
		diff_symbol=None,
	):
		cfg_rescale = 0.75
		cfg_strength = cfg_strength_gop
		temperature = 0
		# deduce batch_size
		if text_list:
			device = text_list[0].device
			batch_size = len(text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list)
   
		assert self.config.experimental.token_dropout_error == 0 and self.config.experimental.token_dropout_rate == 0
		assert total_levels == resps_list[0].shape[-1]
		min_length = 1
		max_length = 5000

		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]
		len_list = [ clamp(resps.shape[0], min_length, max_length) for resps in resps_list]
		gop_list = []
		gop_diff_list = []
		##iterations
		iterator = trange(total_levels, desc="NAR")
		for n in iterator:
			level = n
			quant_levels = [ level for i in range(batch_size)]
			##we constrain it here although in base model, it only sums up untill quant_level as resp embeddings, 
			# provide all the resps up to the current level
			resps_list_in = [ r[..., :level+1] for r in resps_list] 
			
			# full condition section
			inputs = super().inputs(
				text_list=text_list[:1],
				proms_list=proms_list[:1],
				resps_list=resps_list_in[:1],
				lang_list=lang_list[:1],
				is_nar_level_0 = True,
				quant_levels=quant_levels[:1],
			)
			output = super().forward(
				inputs=inputs,
			) 
			#pdb.set_trace()
			#logits = output.logits
			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text[:1],
					proms_list=null_prom[:1],
					resps_list=resps_list_in[:1],
					lang_list=lang_list[:1],
					is_nar_level_0 = True,
					quant_levels=quant_levels[:1],
				)
				null_output = super().forward(
					inputs=null_inputs,
				)

				logits = cfg_logits_modified( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )
					
			else:
				logits = output.logits
			gop_frames = compute_gop(logits[0], resps_list_in[0][:,-1], device)	
			## diff section
			assert diff_symbol is not None
			if diff_symbol == "null":
				diff_phns_list = [torch.tensor([1, 2], device=device)] * batch_size
			else:
				diff_phns = torch.ones_like(text_list[0], device=device) * diff_symbol 
				diff_phns[0] = 1
				diff_phns[-1] = 2
				diff_phns_list = [diff_phns] * batch_size
				
			inputs_diff = super().inputs(
				text_list=diff_phns_list[:1],
				proms_list=proms_list[:1],
				resps_list=resps_list_in[:1],
				lang_list=lang_list[:1],
				is_nar_level_0 = True,
				quant_levels=quant_levels[:1],
			)
			output_diff = super().forward(
				inputs=inputs_diff,
			)

			#logits_diff = output_diff.logits
			if cfg_strength > 0:
				null_inputs_diff = super().inputs(
					text_list=null_text[:1],
					proms_list=null_prom[:1],
					resps_list=resps_list_in[:1],
					lang_list=lang_list[:1],
					is_nar_level_0 = True,
					quant_levels=quant_levels[:1],
				)
				null_output_diff = super().forward(
					inputs=null_inputs_diff,
				)

				logits_diff = cfg_logits_modified( logits=output_diff.logits, null=null_output_diff.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )
			else:
				logits_diff = output_diff.logits
    
			gop_frames_diff = compute_gop(logits_diff[0], resps_list_in[0][:,-1], device)
   
			gop_list.append([gop_frames]*batch_size)
			gop_diff_list.append([gop_frames_diff]*batch_size)
   
		return gop_list,gop_diff_list
  
	def forward_train(
		self,
		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,
	):
		# deduce batch_size
		if text_list:
			device = text_list[0].device
			batch_size = len(text_list)
		elif raw_text_list:
			device = raw_text_list[0].device
			batch_size = len(raw_text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list)

		# specifies how to sample probabilities of which RVQ levels to train against
		rvq_levels_p = self.config.experimental.rvq_levels_p if self.config is not None else "equal"
		# determines which RVQ level to target per batch
		quant_level_range = self.config.experimental.rvq_level_range if self.config is not None and self.config.experimental.rvq_level_range else [ 0 if self.causal else 1, self.n_resp_levels - 1 ]
		# rate to perform token dropout errors
		token_dropout_error = self.config.experimental.token_dropout_error
		# RVQ levels to apply token dropout on
		token_dropout_rvq_levels = self.config.experimental.token_dropout_rvq_levels
		# RVQ levels to apply masking training on
		masking_train_rvq_levels = self.config.experimental.masking_train_rvq_levels  ## always [0,0] for now, for the first level of pure-NAR!!!!
		# CFG (classifier-free guidance?)
		cfg_text_dropout_p = self.config.experimental.cfg_text_dropout_p if self.config is not None else 0.0
		cfg_cond_dropout_p = self.config.experimental.cfg_cond_dropout_p if self.config is not None else 0.0
		cfg_prom_dropout_p = self.config.experimental.cfg_prom_dropout_p if self.config is not None else 0.0
		use_raw_text_p = self.config.experimental.use_raw_text_p if self.config is not None else 0.0
		# rate to train RVQ level AR-ly or NAR-ly
		masking_train_p = self.config.experimental.masking_train_p if self.config is not None else 0.5
		masking_ratio = self.config.experimental.masking_ratio if self.config is not None else "random"
		# force set mask training
		if "len" not in self.capabilities:
			masking_train_p = 0.0
		elif "ar" not in self.capabilities:
			masking_train_p = 1.0
		# implicitly set it to all levels
		if not token_dropout_rvq_levels:
			token_dropout_rvq_levels = [0, self.resp_levels - 1]
		if not token_dropout_rvq_levels:
			token_dropout_rvq_levels = [0, 0]

		# allow passing a specific distribution of RVQ levels
		rvq_levels_p = rvq_levels_p if isinstance(rvq_levels_p, list) else []
		if not rvq_levels_p:
			lo, hi = quant_level_range[0], quant_level_range[1] + 1
			# randomly select a target RVQ-bin level (0 being AR, 1+ being NAR)
			if rvq_levels_p == "equal":
				rvq_levels_p = [ i for i in range( lo, hi ) ]
			else:
				# yuck
				rvq_levels_p = sum([[i for _ in range(hi - i)] for i in range( lo, hi ) ], [])

		# input RVQ levels
		quant_levels = [ random.choice( rvq_levels_p ) for i in range(batch_size) ]
		# timestep levels (for TTS NAR) -- mask schedual?
		timesteps = [ None for _ in range(batch_size) ]

		for i, task in enumerate( task_list ):
			lo, hi = masking_train_rvq_levels[0], masking_train_rvq_levels[1] ## usually [0,0], only on level 0
			if task in text_task:
				quant_levels[i] = 0 # self.n_resp_levels - 1
			elif lo <= quant_levels[i] and quant_levels[i] <= hi and random.random() < masking_train_p:
				# to-do: prioritize lower timesteps over later timesteps
				# ...except that the masking rate is still tied to the cosine scheduling, which does this already
				#r = random.random()
				#p = math.acos(r) / (math.pi * 0.5)
				#timesteps[i] = 1.0 - clamp(p, 0.0, 1.0)
				timesteps[i] = random.random()
				
				# instead make it between [0.2, 0.8]
				if masking_ratio == "rand":
					timesteps[i] = (timesteps[i] * 0.6) + 0.2

		# trim resps to only contain all levels below the target level
		resps_list = [r if t in text_task else r[..., :l+1] for r, l, t in zip(resps_list, quant_levels, task_list)]

		# tensor to cat for RVQ level 0
		text_stop_sequence = torch.tensor([2], device=device, dtype=torch.int16)
		text_start_stop_sequence = torch.tensor([1, 2], device=device, dtype=torch.int16)
		audio_stop_sequence = torch.tensor([[self.stop_token]], device=device, dtype=torch.int16)

		# final validations and stuff
		for i, quant_level, resps, proms, task in zip(range(batch_size), quant_levels, resps_list, proms_list, task_list):
			# cap quant_level if it exceeds its corresponding resp/prom
			# this was needed for when my DAC-encoded audio was erroneously trimmed to 8 RVQ levels instead of 9
			if quant_level >= resps.shape[-1]:
				quant_levels[i] = resps.shape[-1] - 1

			# proms could be a Tensor, list[Tensor], or None
			if isinstance( proms, torch.Tensor ):
				if quant_level >= proms.shape[-1]:
					quant_levels[i] = proms.shape[-1] - 1

			elif isinstance( proms, list ):
				for j, prom in enumerate( proms ):
					if not isinstance( prom, torch.Tensor ):
						continue
					if quant_level >= prom.shape[-1]:
						quant_levels[i] = prom.shape[-1] - 1

			# apply token dropout error compensation 
   			# the model download set error/rate both to 0	
			if token_dropout_error > 0 and (token_dropout_rvq_levels[0] <= quant_level and quant_level <= token_dropout_rvq_levels[1]):
				steps = resps.shape[0]
				for l in range( quant_level ):
					for t in range( steps ):
						token = resps[t, l].item()

						if random.random() < token_dropout_error:								
							offset = 1 * ( 1 if random.random() < 0.5  else -1 )
							resps_list[i][t, l] = clamp(token + offset, 1, 1022) # +- 1

			# only apply stop token for RVQ level 0
			if quant_level <= 0 and timesteps[i] is None:
				# append stop tokens for AR
				if task not in text_task:
					resps_list[i] = torch.cat([ resps, audio_stop_sequence ])

			if task == "len":
				quant_levels[i] = 0

			# apply CFG (should probably only apply to NAR quant level 0)
			if task not in text_task + ["len"]:
				drop_text = False
				drop_audio = False
				swap_text = False

				if random.random() < cfg_prom_dropout_p:
					drop_audio = True
				
				if random.random() < cfg_cond_dropout_p:
					drop_audio = True
					drop_text = True
				
				if random.random() < use_raw_text_p and raw_text_list[i] is not None:
					swap_text = True

				if drop_text:
					text_list[i] = text_start_stop_sequence

				if drop_audio:
					proms_list[i] = None

				if swap_text and not drop_text:
					text_list[i] = None

		inputs = self.inputs(
			text_list=text_list,
			proms_list=proms_list,
			resps_list=resps_list,
			lang_list=lang_list,
			tone_list=tone_list,
			task_list=task_list,
			raw_text_list=raw_text_list,
			time_list=timesteps,

			quant_levels=quant_levels,
		)

		return super().forward(
			inputs=inputs,
			quant_levels=quant_levels,
		)

	## used for resemble diffusion?
	def forward_nar_masked(
		self,

		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,

		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		device = text_list[0].device
		batch_size = len(text_list)

		level = 0  ## can be used for other levels too?
		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

		"""
		def log(t, eps=1e-10):
			return torch.log(t + eps)
		def gumbel_noise(t):
			noise = torch.zeros_like(t).uniform_(0, 1)
			return -log(-log(noise))
		def gumbel_sample(t, temperature=1.0, dim=-1):
			return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)
		"""

		# convert (N)AR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "ar_" )

		min_length = sampling_kwargs.pop("min_duration", 1)
		max_length = sampling_kwargs.pop("max_duration", 500)
		max_steps = sampling_kwargs.get("max_steps", 25)
		refine_on_stop = sampling_kwargs.get("refine_on_stop", False)
		entropix_sampling = sampling_kwargs.get("entropix_sampling", False)
		annealed_sampling = sampling_kwargs.get("annealed_sampling", True)

		# greedy sampling is very, very much preferred, but using greedy logit scores later helps enough
		temperature = sampling_kwargs.pop("temperature", 0.0)
		minimum_cfg_strength = sampling_kwargs.get("minimum_cfg_strength", 2.5)
		# this really helps keep audio coherent so far
		cfg_strength = sampling_kwargs.get("cfg_strength", minimum_cfg_strength)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.75)
		start_noise = sampling_kwargs.get("denoise_start", 0.0)
		end_noise = sampling_kwargs.get("denoise_end", 1.0)
		remasking = sampling_kwargs.get("remasking", True)
		max_steps = math.floor(max_steps * (end_noise - start_noise)) ### when not fully 0->1?

		# to specify the initial mask used
		vc_list = sampling_kwargs.pop("vc_list", None)
		vc_threshold = sampling_kwargs.pop("vc_threshold", 0.25)
		vc_mask_p = sampling_kwargs.pop("vc_mask_p", 0.25)
		if vc_list is not None:
			vc_list = [ x if x.dim() == 1 else x[:, 0] for x in vc_list ]
			len_list = [ x.shape[0] for x in vc_list ]

		len_list = [ clamp(l, min_length, max_length) for l in len_list ]
		
		# force set CFG because too low / no CFG causes issues
		original_cfg_strength = cfg_strength
		cfg_strength = max( cfg_strength, minimum_cfg_strength )

		prefix_context = sampling_kwargs.get("prefix_context", None)
		# we can get away with just providing a list of resps to prefix later, and it will magically get removed anyways when masking and scoring
		if prefix_context is not None:
			text_list = [ torch.concat([prefix[:-1], text[1:]]) for prefix, text in zip( prefix_context[0], text_list ) ]
			prefix_resps_list = [ resps if resps.dim() == 1 else resps[:, 0] for resps in prefix_context[1] ]

		# if we're denoising from an existing sequence
		if start_noise > 0.0 and resps_list is not None:
			# flatten if needed
			resps_list = [ resps if resps.dim() == 1 else resps[:, 0] for resps in resps_list ]
			# gen masking ratio
			noise_p = math.cos( start_noise * math.pi * 0.5 )
			# generate scoring mask (because the above mask will get masked off per the scores, so we do not need to mask beforehand)
			scores = [ torch.tensor( [ 1.0 if random.random() < noise_p else 0.0 for _ in range( seq_len ) ], dtype=torch.float32, device=device ) for seq_len in len_list ]
		else:
			# fill with masked tokens (even though they get masked anyways)
			resps_list = [ torch.ones((seq_len,), dtype=torch.int16, device=device) * self.stop_token for seq_len in len_list ]
			# fill scores
			scores = [ torch.ones((seq_len,), dtype=torch.float32, device=device) for seq_len in len_list ]

		quant_levels = [ level for _ in range(batch_size) ]
		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		iterator = tqdm(torch.linspace(start_noise, end_noise, max_steps), desc="NAR Masked", disable=disable_tqdm)
		for timestep in iterator:
			# update previous list of tokens
			prev_list = resps_list
			# ramp down over time
			annealing = 1.0 - timestep
			# get noise level, per cosine scheduling   
			noise_p = math.cos( timestep * math.pi * 0.5 )  ##warning, in base model, if config.masking_ratio = 0.8. then in training stage, always mask 0.8, but sampling stil follow the noise defined here !!!!!
			# proportion of tokens to remask
			remask_p = 1.0 / (max_steps * 2) if remasking else 0
			# pick the worst scoring tokens to mask off
			masked_indices = [ score.topk( clamp( int( noise_p * seq_len + remask_p * seq_len ), 1, seq_len), dim=-1 ).indices for score, seq_len in zip(scores, len_list) ]
			# normal masking
			if vc_list is None or timestep >= vc_threshold:
				# mask off inputs
				resps_list = [ resp.scatter(0, indices, self.stop_token) for resp, indices in zip( resps_list, masked_indices ) ]  ### why stop_token here, not "0" as done in the base-model forward call for batch processing ? Isn't the stop_token is for AR?
				# boolean mask
				is_masked = [ resps == self.stop_token for resps in resps_list ]
			else:
				# mask off a random portion of the target
				rand_mask_list = [ torch.rand(mask.shape).to(device=device) < vc_mask_p for mask in vc_list ]
				half_mask_list = [ torch.where( rand_mask, self.stop_token, mask.clone() ) for mask, rand_mask in zip( vc_list, rand_mask_list ) ]
				# always set the last end as masked off because it causes issues
				for i, mask in enumerate(half_mask_list):
					half_mask_list[i][-75:] = self.stop_token
				# 
				# mask off inputs per mask
				resps_list = [ resp.scatter(0, indices, mask) for resp, indices, mask in zip( resps_list, masked_indices, half_mask_list ) ]
				# boolean mask
				is_masked = [ resps == mask for resps, mask in zip( resps_list, half_mask_list ) ]

			# timestep inputs
			time_list = [ timestep for _ in range(batch_size) ]

			sampling_temperature = temperature * annealing if annealed_sampling else temperature
			sampling_cfg = cfg_strength * timestep if annealed_sampling else cfg_strength

			# avoid useless CFG sampling
			"""
			if sampling_cfg < minimum_cfg_strength * 0.5:
				sampling_cfg = 0
			"""

			if prefix_context is not None:
				input_resps_list = [ torch.concat( [ prefix, resps ] ) for prefix, resps in zip( prefix_resps_list, resps_list ) ]
				# originally requested no CFG, safe to ignore if we have a prefix
				if original_cfg_strength < minimum_cfg_strength:
					sampling_cfg = original_cfg_strength * timestep if annealed_sampling else original_cfg_strength
			else:
				input_resps_list = resps_list

			# setup inputs
			inputs = super().inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=input_resps_list,
				lang_list=lang_list,
				tone_list=tone_list,
				time_list=time_list,
				quant_levels=quant_levels,
			)
			output = super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
			)

			logits = output.logits

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=input_resps_list,
					lang_list=lang_list,
					tone_list=tone_list,
					time_list=time_list,
					quant_levels=quant_levels,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
				)

				logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )

			# sample with sampler settings
			filtered_sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,

				temperature=sampling_temperature,
				**sampling_kwargs,
			)

			# retrieves unfiltered logits
			unfiltered_sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,

				temperature=0.0,
				**sampling_kwargs,
			)
			# get sampled tokens
			sampled_ids = filtered_sampled.ids
			# keep unmasked tokens
			resps_list = [ torch.where( masked, input_ids, resps ).to(torch.int16) for masked, input_ids, resps in zip( is_masked, sampled_ids, resps_list ) ]
			# get probability scores
			scores = [ 
				# conjugate to have worse scoring tokens picked for topk
				1.0 - 
					# only keep scores of tokens we are predicting (and ignore the tokens previously finalized)
					torch.where( masked, torch.tensor([score for index, score in enumerate(scores)], device=device), torch.ones(masked.shape, device=device) )
				# use unmodified logit scores for this, as it offers better stability
				for scores, masked in zip( unfiltered_sampled.scores, is_masked )
			]

		return resps_list

	## used for resemble diffusion?
	def forward_nar_masked_modified(
		self,

		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,

		disable_tqdm=False,
		use_lora=None,
		n_step_level_0=None,
		is_nar_level_0=None,
		phoneme_mask=None,
		**sampling_kwargs,
	):
		device = resps_list[0].device
		batch_size = len(text_list)
		assert resps_list[0].shape[-1] == 1
		level = 0  ## can be used for other levels too?
		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

		"""
		def log(t, eps=1e-10):
			return torch.log(t + eps)
		def gumbel_noise(t):
			noise = torch.zeros_like(t).uniform_(0, 1)
			return -log(-log(noise))
		def gumbel_sample(t, temperature=1.0, dim=-1):
			return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)
		"""

		# convert (N)AR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "ar_" )

		min_length = sampling_kwargs.pop("min_duration", 1)
		max_length = sampling_kwargs.pop("max_duration", 5000)
		max_steps = sampling_kwargs.get("max_steps", 25)

		refine_on_stop = sampling_kwargs.get("refine_on_stop", False)
		entropix_sampling = sampling_kwargs.get("entropix_sampling", False)
		annealed_sampling = sampling_kwargs.get("annealed_sampling", True)

		# greedy sampling is very, very much preferred, but using greedy logit scores later helps enough
		temperature = sampling_kwargs.pop("temperature", 0.0)
		minimum_cfg_strength = sampling_kwargs.get("minimum_cfg_strength", 2.5)
		# this really helps keep audio coherent so far
		cfg_strength = sampling_kwargs.get("cfg_strength", minimum_cfg_strength)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.75)
  
		# start-end noise
		start_noise_p = phoneme_mask.sum()/resps_list[0].shape[0]
		start_noise = math.acos(start_noise_p)/ (math.pi * 0.5)
		end_noise = 1
		remasking = False
		n_steps = sampling_kwargs.get("n_steps", 25)
		assert n_steps >= 1
		# to specify the initial mask used
		vc_list = sampling_kwargs.pop("vc_list", None)
		prefix_context = sampling_kwargs.get("prefix_context", None)
		if vc_list is not None or prefix_context is not None:
			sys.exit("vc_list is not supprted in this MDD version")
   	
		# force set CFG because too low / no CFG causes issues
		original_cfg_strength = cfg_strength
		cfg_strength = max( cfg_strength, minimum_cfg_strength )
		#cfg_strength = 0

		# masking the input 
		resps_list_in = [torch.where( phoneme_mask, self.stop_token, resps if resps.dim() == 1 else resps[:, 0])[:,None] for resps in resps_list]
		len_list = [ clamp(resps.shape[0], min_length, max_length) for resps in resps_list_in ]
		scores = [ torch.where(phoneme_mask, 1.0, 0.0) for resps in resps_list_in ]
		is_masked = [ resps == self.stop_token for resps in resps_list_in ]
		
		quant_levels = [ level for _ in range(batch_size) ]
		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		## initial generation 
		annealing = 1.0 - start_noise
		sampling_temperature = temperature * annealing if annealed_sampling else temperature
		sampling_cfg = cfg_strength * start_noise if annealed_sampling else cfg_strength
	
		# setup inputs
		inputs = super().inputs(
			text_list=text_list,
			proms_list=proms_list,
			resps_list=resps_list_in,
			lang_list=lang_list,
			tone_list=tone_list,
			quant_levels=quant_levels,
			is_nar_level_0=True, ##False if level!=0 or AR for level 0
			compute_mdd=False,
		)
		output = super().forward(
			inputs=inputs,
			quant_levels=quant_levels,
		)

		logits = output.logits

		if cfg_strength > 0:
			null_inputs = super().inputs(
				text_list=null_text,
				proms_list=null_prom,
				resps_list=resps_list_in,
				lang_list=lang_list,
				tone_list=tone_list,
				quant_levels=quant_levels,
				is_nar_level_0=True, ##False if level!=0 or AR for level 0
				compute_mdd=False,
			)
			null_output = super().forward(
				inputs=null_inputs,
				quant_levels=quant_levels,
			)

			logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )

		# sample with sampler settings
		filtered_sampled = super().sample(
			logits=logits,
			prev_list=resps_list,
			quant_levels=quant_levels,

			temperature=sampling_temperature,
			**sampling_kwargs,
		)

		# retrieves unfiltered logits
		unfiltered_sampled = super().sample(
			logits=logits,
			prev_list=resps_list,
			quant_levels=quant_levels,

			temperature=0.0,
			**sampling_kwargs,
		)
		# get sampled tokens
		sampled_ids = filtered_sampled.ids
		# keep unmasked tokens
		resps_list_out = [ torch.where( masked[:,0], input_ids, resps[:,0] ).to(torch.int16) for masked, input_ids, resps in zip( is_masked, sampled_ids, resps_list ) ]
		# get probability scores
		scores = [ 
			# conjugate to have worse scoring tokens picked for topk
			1.0 - 
				# only keep scores of tokens we are predicting (and ignore the tokens previously finalized)
				torch.where( masked[:,0], torch.tensor(scores, device=device), torch.ones(masked[:,0].shape, device=device) )
			# use unmodified logit scores for this, as it offers better stability
			for scores, masked in zip( unfiltered_sampled.scores, is_masked )
		]
		if n_steps == 1:
			return resps_list_out,logits
		else:
			iterator = tqdm(torch.linspace(start_noise, end_noise, n_steps)[1:], desc="NAR Masked gen", disable=disable_tqdm)
			for timestep in iterator:
				resps_list_in = resps_list_out
				# update previous list of tokens
				prev_list = resps_list_in
				# ramp down over time
				annealing = 1.0 - timestep
				# get noise level, per cosine scheduling   
				noise_p = math.cos( timestep * math.pi * 0.5 )  ##warning, in base model, if config.masking_ratio = 0.8. then in training stage, always mask 0.8, but sampling stil follow the noise defined here !!!!!
				# proportion of tokens to remask
				remask_p = 1.0 / (max_steps * 2) if remasking else 0
				# pick the worst scoring tokens to mask off
				masked_indices = [ score.topk( clamp( int( noise_p * seq_len + remask_p * seq_len ), 1, seq_len), dim=-1 ).indices for score, seq_len in zip(scores, len_list) ]
				# normal masking
				resps_list_in = [ resp.scatter(0, indices, self.stop_token) for resp, indices in zip( resps_list_in, masked_indices ) ]  ### why stop_token here, not "0" as done in the base-model forward call for batch processing ? Isn't the stop_token is for AR?
				# boolean mask
				is_masked = [ resps == self.stop_token for resps in resps_list_in ]

				sampling_temperature = temperature * annealing if annealed_sampling else temperature
				sampling_cfg = cfg_strength * timestep if annealed_sampling else cfg_strength

				# setup inputs
				inputs = super().inputs(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list_in,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
					is_nar_level_0=True, ##False if level!=0 or AR for level 0
					compute_mdd=False,
				)
				output = super().forward(
					inputs=inputs,
					quant_levels=quant_levels,
				)

				logits = output.logits

				if cfg_strength > 0:
					null_inputs = super().inputs(
						text_list=null_text,
						proms_list=null_prom,
						resps_list=resps_list_in,
						lang_list=lang_list,
						tone_list=tone_list,
						quant_levels=quant_levels,
						is_nar_level_0=True, ##False if level!=0 or AR for level 0
						compute_mdd=False,
					)
					null_output = super().forward(
						inputs=null_inputs,
						quant_levels=quant_levels,
					)

					logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )

				# sample with sampler settings
				filtered_sampled = super().sample(
					logits=logits,
					prev_list=prev_list,
					quant_levels=quant_levels,

					temperature=sampling_temperature,
					**sampling_kwargs,
				)

				# retrieves unfiltered logits
				unfiltered_sampled = super().sample(
					logits=logits,
					prev_list=prev_list,
					quant_levels=quant_levels,

					temperature=0.0,
					**sampling_kwargs,
				)
				# get sampled tokens
				sampled_ids = filtered_sampled.ids
				# keep unmasked tokens
				resps_list_out = [ torch.where( masked, input_ids, resps ).to(torch.int16) for masked, input_ids, resps in zip( is_masked, sampled_ids, prev_list ) ]
				# get probability scores
				scores = [ 
					# conjugate to have worse scoring tokens picked for topk
					1.0 - 
						# only keep scores of tokens we are predicting (and ignore the tokens previously finalized)
						torch.where( masked, torch.tensor([score for index, score in enumerate(scores)], device=device), torch.ones(masked.shape, device=device) )
					# use unmodified logit scores for this, as it offers better stability
					for scores, masked in zip( unfiltered_sampled.scores, is_masked )
				]

			return resps_list_out,logits

	## unmaksed for plotting lv0
	def forward_nar_unmasked(
		self,

		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,

		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		device = text_list[0].device
		batch_size = len(text_list)

		level = 0  ## can be used for other levels too?
		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

		"""
		def log(t, eps=1e-10):
			return torch.log(t + eps)
		def gumbel_noise(t):
			noise = torch.zeros_like(t).uniform_(0, 1)
			return -log(-log(noise))
		def gumbel_sample(t, temperature=1.0, dim=-1):
			return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)
		"""

		# convert (N)AR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "ar_" )

		min_length = sampling_kwargs.pop("min_duration", 1)
		max_length = sampling_kwargs.pop("max_duration", 500)
		max_steps = sampling_kwargs.get("max_steps", 25)
		## here only one step for plotting
		assert max_steps == 1
		refine_on_stop = sampling_kwargs.get("refine_on_stop", False)
		entropix_sampling = sampling_kwargs.get("entropix_sampling", False)
		annealed_sampling = sampling_kwargs.get("annealed_sampling", True)

		# greedy sampling is very, very much preferred, but using greedy logit scores later helps enough
		temperature = sampling_kwargs.pop("temperature", 0.0)
		minimum_cfg_strength = sampling_kwargs.get("minimum_cfg_strength", 2.5)
		# this really helps keep audio coherent so far
		cfg_strength = sampling_kwargs.get("cfg_strength", minimum_cfg_strength)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.75)
		# we condition it on everything, so no noise at all
		start_noise = 1
		end_noise = 1
		remasking = False
		#max_steps = math.floor(max_steps * (end_noise - start_noise))

		# to specify the initial mask used
		vc_list = sampling_kwargs.pop("vc_list", None)
		prefix_context = sampling_kwargs.get("prefix_context", None)
		if vc_list is not None or prefix_context is not None:
			sys.exit("vc_list is not supprted in this MDD version")
		
		# force set CFG because too low / no CFG causes issues
		original_cfg_strength = cfg_strength
		cfg_strength = max( cfg_strength, minimum_cfg_strength )

		len_list = [ clamp(resps.shape[0], min_length, max_length) for resps in resps_list ]
		annealing = 1.0 - start_noise
		sampling_temperature = temperature * annealing if annealed_sampling else temperature
  
		quant_levels = [ level for _ in range(batch_size) ]
		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		## generate only once
		resps_list_in = resps_list
		len_list = [ clamp(resps.shape[0], min_length, max_length) for resps in resps_list_in ]
		annealing = 1.0 - start_noise
		sampling_temperature = temperature * annealing if annealed_sampling else temperature
  
		# setup inputs
		inputs = super().inputs(
			text_list=text_list,
			proms_list=proms_list,
			resps_list=resps_list_in,
			lang_list=lang_list,
			tone_list=tone_list,
			quant_levels=quant_levels,
			is_nar_level_0=True, ##False if level!=0 or AR for level 0
			compute_mdd=False,
		)
		output = super().forward(
			inputs=inputs,
			quant_levels=quant_levels,
		)

		logits = output.logits

		if cfg_strength > 0:
			null_inputs = super().inputs(
				text_list=null_text,
				proms_list=null_prom,
				resps_list=resps_list_in,
				lang_list=lang_list,
				tone_list=tone_list,
				quant_levels=quant_levels,
				is_nar_level_0=True, ##False if level!=0 or AR for level 0
				compute_mdd=False,
			)
			null_output = super().forward(
				inputs=null_inputs,
				quant_levels=quant_levels,
			)

			logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ l for l in len_list ] )

		# sample with sampler settings
		filtered_sampled = super().sample(
			logits=logits,
			prev_list=resps_list,
			quant_levels=quant_levels,

			temperature=sampling_temperature,
			**sampling_kwargs,
		)

		# retrieves unfiltered logits
		unfiltered_sampled = super().sample(
			logits=logits,
			prev_list=resps_list,
			quant_levels=quant_levels,

			temperature=0.0,
			**sampling_kwargs,
		)
		# get sampled tokens
		sampled_ids = filtered_sampled.ids
		# keep unmasked tokens
		resps_list_out = sampled_ids
  
		return resps_list_out,logits
		

	### normal nar?
	def forward_nar(
		self,
		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		
		raw_text_list: list[Tensor] | None = None,

		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		# inference NAR level 0
		if len_list is not None:
			resps_list = self.forward_nar_masked(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				task_list=task_list,
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				**sampling_kwargs,				
			)
		
		# deduce batch_size
		if text_list:
			device = text_list[0].device
			batch_size = len(text_list)
		elif raw_text_list:
			device = raw_text_list[0].device
			batch_size = len(raw_text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list)
		
		# convert NAR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "nar_" )

		max_levels = sampling_kwargs.get("max_levels", 0)
		cfg_strength = sampling_kwargs.get("cfg_strength", 0.0)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.7)

		if max_levels == 0:
			max_levels = self.n_max_levels - 1

		# expand if given a raw 1D tensor
		for i, resp in enumerate(resps_list):
			if resp.dim() == 1:
				resps_list[i] = resp.unsqueeze(-1)
		
		prev_list = resps_list

		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		iterator = trange( max_levels, desc="NAR", disable=disable_tqdm )
		for n in iterator:
			level = prev_list[0].shape[-1]
			if level >= max_levels + 1:
				iterator.close()
				break

			if cfg.lora is not None:
				enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

			quant_levels = [ level for _ in range(batch_size) ]

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=prev_list,  ### it is a BXTXL Tensor 0<L<8, codeword not embeddings! 
				lang_list=lang_list,
				tone_list=tone_list,
				quant_levels=quant_levels,
			)

			output = super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
			)
			logits, state = output.logits, output.state

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=prev_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
				)
				## rememebered in probAI, CFG = conditional + scale * unconditional(null_input)
				logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ resp.shape[0] for resp in resps_list ] )

			sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,
				**(sampling_kwargs),
			)

			resps_list = sampled.ids
			## always concate, because next step need all the preivious code-levels for embeddings ! different from original
			prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device=device)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

		return prev_list

 
	### fixed_generation
	def forward_fixed_generation(
		self,
		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		
		raw_text_list: list[Tensor] | None = None,
		fix_level = None,
		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		if fix_level is None:
			sys.exit("must specify a fix level")
		assert fix_level + 1 == resps_list[0].shape[-1]
		assert len_list == [ resp.shape[0] for resp in resps_list]
	
		# deduce batch_size
		if resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list)
		elif raw_text_list:
			device = raw_text_list[0].device
			batch_size = len(raw_text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif text_list:
			device = text_list[0].device
			batch_size = len(text_list)
		
		# convert NAR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "nar_" )

		max_levels = sampling_kwargs.get("max_levels", 0)
		cfg_strength = sampling_kwargs.get("cfg_strength", 0.0)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.7)

		if max_levels == 0:
			max_levels = self.n_max_levels - 1

		# expand if given a raw 1D tensor
		for i, resp in enumerate(resps_list):
			if resp.dim() == 1:
				resps_list[i] = resp.unsqueeze(-1)
		
		prev_list = resps_list

		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		iterator = trange( max_levels, desc="NAR", disable=disable_tqdm )
		for n in iterator:
			level = prev_list[0].shape[-1]
			if level >= max_levels + 1:
				iterator.close()
				break

			if cfg.lora is not None:
				enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

			quant_levels = [ level for _ in range(batch_size) ]

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=prev_list,  ### it is a BXTXL Tensor 0<L<8, codeword not embeddings! 
				lang_list=lang_list,
				tone_list=tone_list,
				quant_levels=quant_levels,
				compute_mdd = False,
			)

			output = super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
			)
			logits, state = output.logits, output.state

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=prev_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
					compute_mdd = False,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
				)
				## rememebered in probAI, CFG = conditional + scale * unconditional(null_input)
				logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ resp.shape[0] for resp in resps_list ] )

			sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,
				**(sampling_kwargs),
			)

			resps_list = sampled.ids
			## always concate, because next step need all the preivious code-levels for embeddings ! different from original
			prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device=device)], dim=-1) for rs, r in zip(prev_list, resps_list) ]

		return prev_list

	### masked_generation
	def forward_masked_generation(
		self,
		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		
		raw_text_list: list[Tensor] | None = None,
		predict_level_0=None,
  		phoneme_mask=None,
    	n_step_level_0=None,
		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		if len(resps_list) != 1:
			sys.exit("masked generation only once at a time")
		if resps_list[0].shape[-1] != 1 :
			sys.exit("masked generation needs only one level of code")
		# deduce batch_size
		# if text_list:
		# 	device = text_list[0].device
		# 	batch_size = len(text_list)
		# elif raw_text_list:
		# 	device = raw_text_list[0].device
		# 	batch_size = len(raw_text_list)
		# elif proms_list:
		# 	device = proms_list[0].device
		# 	batch_size = len(proms_list)
		# elif resps_list:
		# 	device = resps_list[0].device
		# 	batch_size = len(resps_list)	
		device = resps_list[0].device
		batch_size = 1
  
		assert phoneme_mask.shape[0] == resps_list[0].shape[0]
		assert len_list == [ resp.shape[0] for resp in resps_list]
		logits_list = []
	
		# convert NAR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "nar_" )

		max_levels = sampling_kwargs.get("max_levels", 0)
		cfg_strength = sampling_kwargs.get("cfg_strength", 0.0)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.7)

		if max_levels == 0:
			max_levels = self.n_max_levels - 1

		# expand if given a raw 1D tensor
		for i, resp in enumerate(resps_list):
			if resp.dim() == 1:
				resps_list[i] = resp.unsqueeze(-1)
		
		if predict_level_0:
		## provide the phoneme_mask to base class for NAR-level-0 masked generation
			assert n_step_level_0 is not None
			mask_token = self.stop_token
			#quant_levels = [ 0 for i in range(batch_size)]
			##ecker said that level0 sampling using greedy always (temperature = 0)
			sampling_kwargs_level_0 = {"n_steps":n_step_level_0}

			prev_list, nar_logits = self.forward_nar_masked_modified(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				lang_list=lang_list,
				is_nar_level_0=True, ##False if level!=0 or AR for level 0
				compute_mdd=False,
				phoneme_mask=phoneme_mask,
				**sampling_kwargs_level_0,
			)
			prev_list = [prev_code[:,None] for prev_code in prev_list]
			logits_list.append(nar_logits)
		else: 
        ## skipping predicting level 0 masked part. In this case, we do the masking here
			## we can only use random or constant 0 here becasue only NAR:0:0 has stop_token in the embedding
			mask_token = 0 
			prev_code = torch.where( phoneme_mask, mask_token, resps_list[0] if resps_list[0].dim() == 1 else resps_list[0][:, 0])
			prev_list = [prev_code[:,None]]
   
		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		iterator = trange( max_levels, desc="NAR", disable=disable_tqdm )
		for n in iterator:
			level = prev_list[0].shape[-1]
			if level >= max_levels + 1:
				iterator.close()
				break

			if cfg.lora is not None:
				enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

			quant_levels = [ level for _ in range(batch_size) ]

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=prev_list,  ### it is a BXTXL Tensor 0<L<8, codeword not embeddings! 
				lang_list=lang_list,
				tone_list=tone_list,
				quant_levels=quant_levels,
				compute_mdd=False,
			)

			output = super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
			)
			logits, state = output.logits, output.state

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=prev_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
					compute_mdd = False,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
				)
				## rememebered in probAI, CFG = conditional + scale * unconditional(null_input)
				logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ resp.shape[0] for resp in resps_list ] )
			
			logits_list.append(logits)
			sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,
				**(sampling_kwargs),
			)

			resps_list_out = sampled.ids
			## always concate, because next step need all the preivious code-levels for embeddings ! different from original
			prev_list = [ torch.cat([rs, r.unsqueeze(-1).to(device=device)], dim=-1) for rs, r in zip(prev_list, resps_list_out) ]

		return prev_list,logits_list 



	### masked for plotting, no generation
	def forward_masked_for_plotting(
		self,
		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		
		raw_text_list: list[Tensor] | None = None,
		predict_level_0=None,
  		phoneme_mask=None,
    	n_step_level_0=None,
		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		if len(resps_list) != 1:
			sys.exit("masked plotting only once at a time")
		if resps_list[0].shape[-1] != 8:
			sys.exit("masked plotting needs all level of codes")
		# deduce batch_size
		# if text_list:
		# 	device = text_list[0].device
		# 	batch_size = len(text_list)
		# elif raw_text_list:
		# 	device = raw_text_list[0].device
		# 	batch_size = len(raw_text_list)
		# elif proms_list:
		# 	device = proms_list[0].device
		# 	batch_size = len(proms_list)
		# elif resps_list:
		# 	device = resps_list[0].device
		# 	batch_size = len(resps_list)	
		device = resps_list[0].device
		batch_size = 1
  
		assert phoneme_mask.shape[0] == resps_list[0].shape[0]
		assert len_list == [ resp.shape[0] for resp in resps_list]
		logits_list = []
	
		# convert NAR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "nar_" )

		max_levels = sampling_kwargs.get("max_levels", 0)
		cfg_strength = sampling_kwargs.get("cfg_strength", 0.0)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.7)

		if max_levels == 0:
			max_levels = self.n_max_levels - 1

		# expand if given a raw 1D tensor
		for i, resp in enumerate(resps_list):
			if resp.dim() == 1:
				resps_list[i] = resp.unsqueeze(-1)
		
		if predict_level_0:
		## provide the phoneme_mask to base class for NAR-level-0 masked generation
			assert n_step_level_0 is not None
			mask_token = self.stop_token
			#quant_levels = [ 0 for i in range(batch_size)]
			##ecker said that level0 sampling using greedy always (temperature = 0)
			sampling_kwargs_level_0 = {"n_steps":n_step_level_0}

			prev_list, nar_logits = self.forward_nar_masked_modified(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=[resps[:,:1] for resps in resps_list],
				lang_list=lang_list,
				is_nar_level_0=True, ##False if level!=0 or AR for level 0
				compute_mdd=False,
				phoneme_mask=phoneme_mask,
				**sampling_kwargs_level_0,
			)
			resps_list_out = [prev_code[:,None] for prev_code in prev_list]
			logits_list.append(nar_logits)
		else: 
        ## skipping predicting level 0 masked part. In this case, we do the masking here
			## we can only use random or constant 0 here becasue only NAR:0:0 has stop_token in the embedding
			mask_token = 0 
			prev_code = torch.where( phoneme_mask, mask_token, resps_list[0] if resps_list[0].dim() == 1 else resps_list[0][:, 0])
			prev_list = [prev_code[:,None]]
   
		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		iterator = trange( max_levels, desc="NAR", disable=disable_tqdm )
		for n in iterator:
			##rewrite previous list with input
			prev_list = [ resps[:, :n+1] for resps in resps_list]
			level = prev_list[0].shape[-1]
			if level >= max_levels + 1:
				iterator.close()
				break

			if cfg.lora is not None:
				enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

			quant_levels = [ level for _ in range(batch_size) ]

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=prev_list,  ### it is a BXTXL Tensor 0<L<8, codeword not embeddings! 
				lang_list=lang_list,
				tone_list=tone_list,
				quant_levels=quant_levels,
				compute_mdd=False,
			)

			output = super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
			)
			logits, state = output.logits, output.state

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=prev_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
					compute_mdd = False,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
				)
				## rememebered in probAI, CFG = conditional + scale * unconditional(null_input)
				logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ resp.shape[0] for resp in resps_list ] )
			
			logits_list.append(logits)
			sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,
				**(sampling_kwargs),
			)

			resps_list_temp = sampled.ids
			## always concate, because next step need all the preivious code-levels for embeddings ! different from original
			resps_list_out = [ torch.cat([rs, r.unsqueeze(-1).to(device=device)], dim=-1) for rs, r in zip(resps_list_out, resps_list_temp) ]

		return resps_list_out,logits_list 

	### unmask for plotting, no generation
	def forward_unmasked_for_plotting(
		self,
		task_list: list[Tensor] | None = None,
		
		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		
		raw_text_list: list[Tensor] | None = None,
		predict_level_0=None,
    	n_step_level_0=None,
		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		if len(resps_list) != 1:
			sys.exit("masked plotting only once at a time")
		if resps_list[0].shape[-1] != 8:
			sys.exit("masked plotting needs all level of codes")


		device = resps_list[0].device
		batch_size = 1
		assert len_list == [ resp.shape[0] for resp in resps_list]
		logits_list = []
	
		# convert NAR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "nar_" )

		max_levels = sampling_kwargs.get("max_levels", 0)
		cfg_strength = sampling_kwargs.get("cfg_strength", 0.0)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.7)

		if max_levels == 0:
			max_levels = self.n_max_levels - 1

		# expand if given a raw 1D tensor
		for i, resp in enumerate(resps_list):
			if resp.dim() == 1:
				resps_list[i] = resp.unsqueeze(-1)
		
		assert predict_level_0 == True
		##unmasked, conditioned on all input, so no need of denoising
		assert n_step_level_0 == 1
		mask_token = self.stop_token
		#quant_levels = [ 0 for i in range(batch_size)]
		##ecker said that level0 sampling using greedy always (temperature = 0)
		sampling_kwargs_level_0 = {"max_steps":n_step_level_0}

		prev_list, nar_logits = self.forward_nar_unmasked(
			text_list=text_list,
			proms_list=proms_list,
			resps_list=[resps[:,:1] for resps in resps_list],
			lang_list=lang_list,
			is_nar_level_0=True, ##False if level!=0 or AR for level 0
			**sampling_kwargs_level_0,
		)
		resps_list_out = [prev_code[:,None] for prev_code in prev_list]
		logits_list.append(nar_logits)
   
		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		iterator = trange( max_levels, desc="NAR", disable=disable_tqdm )
		for n in iterator:
			##rewrite previous list with input
			prev_list = [ resps[:, :n+1] for resps in resps_list]
			level = prev_list[0].shape[-1]
			if level >= max_levels + 1:
				iterator.close()
				break

			if cfg.lora is not None:
				enable_lora( self, cfg.lora.active_level( level ) if use_lora is None else use_lora )

			quant_levels = [ level for _ in range(batch_size) ]

			inputs = self.inputs(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=prev_list,  ### it is a BXTXL Tensor 0<L<8, codeword not embeddings! 
				lang_list=lang_list,
				tone_list=tone_list,
				quant_levels=quant_levels,
				compute_mdd=False,
			)

			output = super().forward(
				inputs=inputs,
				quant_levels=quant_levels,
			)
			logits, state = output.logits, output.state

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=prev_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
					compute_mdd = False,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
				)
				## rememebered in probAI, CFG = conditional + scale * unconditional(null_input)
				logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ resp.shape[0] for resp in resps_list ] )
			
			logits_list.append(logits)
			sampled = super().sample(
				logits=logits,
				prev_list=prev_list,
				quant_levels=quant_levels,
				**(sampling_kwargs),
			)

			resps_list_temp = sampled.ids
			## always concate, because next step need all the preivious code-levels for embeddings ! different from original
			resps_list_out = [ torch.cat([rs, r.unsqueeze(-1).to(device=device)], dim=-1) for rs, r in zip(resps_list_out, resps_list_temp) ]

		return resps_list_out,logits_list 

	### can be used for predicting len for NAR or AR for level 0, can be used for other levels as well
	def forward_ar(
		self,

		task_list: list[Tensor],

		text_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,

		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		# deduce batch_size
		if text_list:
			device = text_list[0].device
			batch_size = len(text_list)
		elif raw_text_list:
			device = raw_text_list[0].device
			batch_size = len(raw_text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list)

		if cfg.lora is not None:
			enable_lora( self, cfg.lora.active_level( 0 ) if use_lora is None else use_lora )

		# convert AR specific args
		sampling_kwargs = convert_kwargs( sampling_kwargs, "ar_" )

		temperature = sampling_kwargs.get("temperature", 1.0)
		cfg_strength = sampling_kwargs.get("cfg_strength", 0.0)
		cfg_rescale = sampling_kwargs.pop("cfg_rescale", 0.7)
		min_temperature = sampling_kwargs.get("min_temperature", -1.0)
		max_duration = sampling_kwargs.get("max_duration", 500)
		beam_width = sampling_kwargs.get("beam_width", 0)
		entropix_sampling = sampling_kwargs.get("entropix_sampling", False)
		refine_on_stop = sampling_kwargs.get("refine_on_stop", False)
		input_prompt_prefix = sampling_kwargs.get("input_prompt_prefix", False)
		layer_skip = sampling_kwargs.get("layer_skip", False)
		prefix_silence = sampling_kwargs.get("prefix_silence", 0.0)
		mirostat_tau = sampling_kwargs.get("mirostat_tau", 0.0)
		mirostat_eta = sampling_kwargs.get("mirostat_eta", 0.0)

		# inference len
		if task_list is not None and task_list[0] == "len":
			sequence_list = [ torch.tensor([0], device=device,dtype=torch.int16) for _ in range(batch_size) ]
			stopped = torch.zeros(batch_size, device=device).bool()
			
			stop_token = 10
			task_list = [ "len" for _ in range(batch_size) ]
			quant_levels = [ 0 for _ in range( max( batch_size, beam_width ) ) ]

			iterator = trange(10, desc="AR", disable=disable_tqdm)  ## at most length = 99999999? AR-attention generates each time step one token 
			for n in iterator:
				len_list = sequence_list

				inputs = self.inputs(
					task_list=task_list,
					
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list,
					
					lang_list=lang_list,
					tone_list=tone_list,
					len_list=len_list,
					raw_text_list=raw_text_list,
					
					quant_levels=quant_levels,
				)

				output = super().forward(
					inputs=inputs, 
					quant_levels=quant_levels,  ### every call to forward can generate logits for only one level, important!
				)
				logits = output.logits #(b,t,v)  

				r = [ logit[-1:].argmax(dim=1) for logit in logits ]  ### forward() will remove the logits for padding-masking and causal-masking,so the last one is the newly generated token
				# sanitize
				for i, token in enumerate(r):
					if token > stop_token:  ## for task "len", only 0-9 is valid?
						r[i][0] = stop_token

				# append tokens -- so the input is the expanded and move to the next step
				for i, ri in enumerate(r):
					if stop_token in ri:
						stopped[i] = True
					sequence_list[i] = torch.cat([sequence_list[i], ri.to(device)])

				# stop token found
				stopped |= r == stop_token
				if stopped.all().item():
					iterator.close()
					break

			# convert tokens into int
			return [ int("".join([ str(token.item()) for token in r if token != stop_token ])) for r in sequence_list ]

		start_slice = [ 0 for _ in range(batch_size) ]
		sequence_list = [ torch.zeros(0, device=device).to(torch.int16) for _ in range(batch_size) ]
		stopped = torch.zeros(batch_size, device=device).bool()
		
		audio_stop_token = self.stop_token
		text_stop_token = 2

		state = None
		mirostat = [
			{"n": 1024, "tau": mirostat_tau, "eta": mirostat_eta, "max_surprise": mirostat_eta * 2, "error_surprise": 0, "running_total_surprise": 0}
		] * batch_size if mirostat_tau > 0.0 else None

		scores = [ 1.0 ] * beam_width
		metrics = []

		"""
		sampling_layer_skip_variables = {} if sampling_layer_skip else None

		if sampling_layer_skip:
			if sampling_layer_skip_entropy_threshold >= 0:
				sampling_layer_skip_variables["entropy_threshold"] = sampling_layer_skip_entropy_threshold
			if sampling_layer_skip_varentropy_threshold >= 0:
				sampling_layer_skip_variables["varentropy_threshold"] = sampling_layer_skip_varentropy_threshold
			if sampling_layer_skip_exit_layer >= 0:
				sampling_layer_skip_variables["max_layer"] = sampling_layer_skip_exit_layer
		"""

		for i, sequence in enumerate( sequence_list ):
			# add <bos> to text for STT
			if task_list[i] in text_task:
				start_slice[i] = 1
				sequence_list[i] = torch.cat([sequence_list[i], torch.tensor([1], dtype=torch.int16, device=device)])
			# treat input prompt as initial resp (by prefixing with the prompt instead)
			elif input_prompt_prefix:
				start_slice[i] = proms_list[i].shape[0]
				sequence_list[i], proms_list[i] = proms_list[i][:, 0], sequence_list[i]
			elif prefix_silence > 0:
				sequence_list[i] = get_silence(prefix_silence, device=sequence_list[i].device)
				sequence_list[i] = sequence_list[i][:, 0]
				# start_slice[i] = sequence_list[i].shape[0]

		# prefixed context provided
		prefix_context = sampling_kwargs.get("prefix_context", None)
		if prefix_context is not None:
			prefix_text, prefix_resps, _ = prefix_context
			# to-do: check if we actually need to drop the middle "<eos><bos>"
			text_list = [ torch.concat([prefix[:-1], text[1:]]) for prefix, text in zip( prefix_text, text_list ) ]
			# feeding this into the NAR-len should automatically handle things
			sequence_list = [ resps if resps.dim() == 1 else resps[:, 0] for resps in prefix_resps ]

		null_text = [ torch.tensor([1, 2], device=device, dtype=torch.int16) for _ in range(batch_size) ]
		null_prom = [ None for _ in range(batch_size) ]

		# get next in sequence
		iterator = trange(max_duration // max(1, self.causal_size), desc="AR", disable=disable_tqdm)
		for n in iterator:
			if batch_size == 1 and task_list[0] in ["phn", "un-phn"]:
				text_list = [ sequence_list[i] if task in ["phn"] else text_list[i] for i, task in enumerate(task_list) ]
				raw_text_list = [ sequence_list[i] if task in ["un-phn"] else raw_text_list[i] for i, task in enumerate(task_list) ]
			else:
				if raw_text_list is not None:
					raw_text_list = [ sequence_list[i] if task in text_task else raw_text_list[i] for i, task in enumerate(task_list) ]
				else:
					text_list = [ sequence_list[i] if task in text_task else text_list[i] for i, task in enumerate(task_list) ]
				resps_list = [ sequence_list[i] if task not in text_task else resps_list[i] for i, task in enumerate(task_list) ]

			quant_levels = [ 0 for _ in range( max( batch_size, beam_width ) ) ]

			inputs = self.inputs(
				task_list=task_list,
				
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,
				
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				raw_text_list=raw_text_list,
				
				quant_levels=quant_levels,
			)

			# to-do: find an elegant way to write this
			output = super().forward(
				inputs=inputs,
				state=state,
				#layer_skip_variables=sampling_layer_skip_variables,
				output_attentions=entropix_sampling,
			)

			if cfg_strength > 0:
				null_inputs = super().inputs(
					text_list=null_text,
					proms_list=null_prom,
					resps_list=resps_list,
					lang_list=lang_list,
					tone_list=tone_list,
					quant_levels=quant_levels,
				)
				null_output = super().forward(
					inputs=null_inputs,
					quant_levels=quant_levels,
					#layer_skip_variables=sampling_layer_skip_variables,
				)
				logits = cfg_logits( logits=output.logits, null=null_output.logits, strength=cfg_strength, rescale=cfg_rescale, lens=[ resp.shape[0] + 1 for resp in resps_list ] )
			
			logits, state = output.logits, output.state

			sampled = super().sample(
				logits=logits,
				prev_list=[ resps_list[i] if task not in text_task else text_list[i] for i, task in enumerate( task_list ) ],
				**(sampling_kwargs | {"attentions": output.attentions if entropix_sampling else None}),
			)

			ids = sampled.ids

			if cfg.experimental:
				if sampled.entropy:
					metrics.append( sampled.entropy )
				elif sampled.scores:
					#metrics.append( [ { "p": p[0], "exited_layer": output.exited_layer } for p in sampled.scores ] )
					metrics.append( [ { "p": p[0] } for p in sampled.scores ] )

			if mirostat is not None:
				mirostat = sampled.scores
			elif beam_width > 0:
				# expand tuple
				s = sampled.scores
				# first step, expand batch
				if batch_size == 1:
					batch_size = beam_width
					text_list = text_list * beam_width
					proms_list = proms_list * beam_width
					sequence_list = sequence_list * beam_width
					task_list = task_list * beam_width
					start_slice = start_slice * beam_width
					stopped = torch.zeros(batch_size, device=device).bool()

				scores = [ scores[i] + score for i, score in enumerate(s) ]

			# append tokens
			for i, token in enumerate(ids):
				task = task_list[i]
				stop_token = audio_stop_token if task not in text_task else text_stop_token
				if stop_token in token:
					stopped[i] = True
				sequence_list[i] = torch.cat([sequence_list[i], token.to(device)])

			# stop token found
			# stopped |= r == stop_token
			if stopped.all().item():
				iterator.close()
				break

		# to-do for layerskip / speculative sampling: rerun the last sequence again at max depth
		"""
		if metrics:
			from ..plot import plot_sample_metrics
			filename = "metrics"
			if entropix_sampling:
				filename += f'[entropix_sampling]'
			if sampling_layer_skip_exit_layer >= 0:
				filename += f'[{sampling_layer_skip_exit_layer+1}]'

			plot_sample_metrics( metrics, filename=f'{filename}.png' )
		"""

		# pick the best scoring candidate
		# desu this is always going to be candidate 0
		if beam_width:
			sequence_list = sequence_list[:1]
			task_list = task_list[:1]

		# remove stop token
		sequence_list = [self._prune(r, audio_stop_token if task_list[i] not in text_task else text_stop_token) for i, r in enumerate(sequence_list)]
		# remove <bos>
		sequence_list = [ sequence_list[i][start_slice[i]:] for i, task in enumerate( task_list ) ]

		if refine_on_stop:
			# get how much we need to slice from the end
			slice_lengths = [ sequence.shape[-1] for sequence in sequence_list ]
			# -1 for the stop token
			logits = [ logit[-length-1:-1] for logit, length in zip(logits, slice_lengths) ]
			# greedy sample from the sequence
			refined_list = [ logit.argmax(dim=-1) for logit in logits ]
			# to-do: compare scores
			# set the "refined" list as the output
			sequence_list = refined_list

		# slice out prefix
		if prefix_context is not None:
			prefix_text, prefix_resps, prefix_lens = prefix_context
			sequence_list = [ resps[l:] for resps, l in zip(sequence_list, prefix_lens) ]

		return sequence_list

	def forward(
		self,
		task_list: list[Tensor] | None = None,

		text_list: list[Tensor] | None = None,
		proms_list: list[Tensor] | None = None,
		resps_list: list[Tensor] | None = None,
		
		lang_list: list[Tensor] | None = None,
		tone_list: list[Tensor] | None = None,
		len_list: list[Tensor] | None = None,
		raw_text_list: list[Tensor] | None = None,
		##general
  		n_step_level_0 = None,
		cfg_strength_lv0 = None,

		##mdd
		total_levels = None,
		is_mdd: bool | None = None,
		phoneme_mask_list = None,
		diff_symbol = None,
		pid_seq = None,
  		is_masking_nar_level_0 = None,
		mask_ratio_lv0 = None,
		##tts
		fix_level=None,
		predict_level_0=None,
		phoneme_mask: Tensor | None=None,
		to_plot=False,
     
		disable_tqdm=False,
		use_lora=None,
		**sampling_kwargs,
	):
		# deduce batch_size
		if resps_list:
			device = resps_list[0].device
			batch_size = len(resps_list) 
		elif raw_text_list:
			device = raw_text_list[0].device
			batch_size = len(raw_text_list)
		elif proms_list:
			device = proms_list[0].device
			batch_size = len(proms_list)
		elif text_list:
			device = text_list[0].device
			batch_size = len(text_list)
   
		# check correct input for MDD
		tts_check = [True if task=="tts" else False for task in task_list ]
		all_tts = all(tts_check)
		if is_mdd and all_tts is not None and text_list is not None and resps_list is not None and proms_list is not None and diff_symbol is not None:
			n_levels_set = {r.shape[-1] for r in resps_list} 
			n_levels = next(iter(n_levels_set))
			assert (n_levels == self.n_resp_levels)  ## resp must full 
			if phoneme_mask_list is not None:
				return self.forward_mdd_nar_v2(
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list,		
					lang_list=lang_list,
					tone_list=tone_list,
		
					total_levels=total_levels,
					cfg_strength_gop=cfg_strength_lv0,
					phoneme_mask_list=phoneme_mask_list,
					diff_symbol = diff_symbol,
					n_step_level_0=n_step_level_0,
		
					disable_tqdm=disable_tqdm,
					use_lora=use_lora,
					**sampling_kwargs,
				)

			elif pid_seq is not None:
				return self.forward_mdd_nar(
				task_list=task_list,
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,			
				lang_list=lang_list,
				tone_list=tone_list,
				raw_text_list=raw_text_list,
				pid_seq=pid_seq,
				is_masking_nar_level_0=is_masking_nar_level_0,
				total_levels=total_levels,
				cfg_strength_gop=cfg_strength_lv0,
				mask_ratio_lv0=mask_ratio_lv0,
				diff_symbol = diff_symbol,
				disable_tqdm=disable_tqdm,
				use_lora=use_lora,
				**sampling_kwargs,
				)
			else:
				return self.forward_mdd_nar_nomask(
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,			
				lang_list=lang_list,
				tone_list=tone_list,
				total_levels=total_levels,
				cfg_strength_gop=cfg_strength_lv0,
				diff_symbol = diff_symbol,
				disable_tqdm=disable_tqdm,
				use_lora=use_lora,
				**sampling_kwargs,
				)
				


		## not mdd meaning for generation
		elif not is_mdd and all_tts and resps_list is not None and fix_level is not None:
	    ##do fixed level generation
			return self.forward_fixed_generation(
				task_list=task_list,
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,			
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				raw_text_list=raw_text_list,
				disable_tqdm=disable_tqdm,
				use_lora=use_lora,
				fix_level=fix_level,
				**sampling_kwargs,
			)
		elif not is_mdd and all_tts and resps_list is not None and phoneme_mask is not None and predict_level_0 is not None and to_plot==False:
		##do masked generation
			return self.forward_masked_generation(
				task_list=task_list,
				text_list=text_list,
				proms_list=proms_list,
				resps_list=resps_list,			
				lang_list=lang_list,
				tone_list=tone_list,
				len_list=len_list,
				raw_text_list=raw_text_list,
				disable_tqdm=disable_tqdm,
				use_lora=use_lora,	
				predict_level_0=predict_level_0,
    			phoneme_mask=phoneme_mask,
				n_step_level_0 = n_step_level_0,
				**sampling_kwargs,
			)
		elif not is_mdd and all_tts and resps_list is not None and predict_level_0 is not None and to_plot==True:
			if phoneme_mask is None:
				##do inference for plotting
				return self.forward_unmasked_for_plotting(
					task_list=task_list,
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list,			
					lang_list=lang_list,
					tone_list=tone_list,
					len_list=len_list,
					raw_text_list=raw_text_list,
					disable_tqdm=disable_tqdm,
					use_lora=use_lora,	
					predict_level_0=predict_level_0,
					n_step_level_0 = n_step_level_0,
					**sampling_kwargs,
				)
			else:
				##do masked inference for plotting
				return self.forward_masked_for_plotting(
					task_list=task_list,
					text_list=text_list,
					proms_list=proms_list,
					resps_list=resps_list,			
					lang_list=lang_list,
					tone_list=tone_list,
					len_list=len_list,
					raw_text_list=raw_text_list,
					disable_tqdm=disable_tqdm,
					use_lora=use_lora,	
					predict_level_0=predict_level_0,
					phoneme_mask=phoneme_mask,
					n_step_level_0 = n_step_level_0,
					**sampling_kwargs,
				)
		else:
			sys.exit("can't not identify the task, check the inputs combinitions")
		### for training, the dataloader/dataset will sample the tasks to train len/nar/ar...., not sure if multiple tasks are allowed for a singel batch?
		### For level 0, Masked NAR / normal NAR training is controled by config, for other levels AR/NAR is controled by using masking for "causal" attention? 		   


# def example_usage():
# 	cfg.device = "cuda"
# 	cfg.trainer.backend = "local"
# 	if cfg.audio_backend == "dac":
# 		cfg.sample_rate = 44_100

# 	from functools import partial
# 	from einops import repeat
# 	from tqdm import tqdm

# 	from ..emb.qnt import decode_to_file, unload_model, trim_random, repeat_extend_audio, concat_audio, merge_audio
# 	from ..engines import Engine, Engines
# 	from ..utils import wrapper as ml
# 	from ..utils import setup_logging
	
# 	import numpy as np
# 	import re
	
# 	# cfg.model.experimental.masking_train_p = 0.5
# 	cfg.hyperparameters.batch_size = 1
# 	cfg.hyperparameters.gradient_accumulation_steps = 1

# 	setup_logging()

# 	def load_artifact( path ):
# 		artifact = np.load(path, allow_pickle=True)[()]

# 		text = torch.tensor( cfg.tokenizer.encode( artifact["metadata"]["phonemes"] ) ).to(dtype=torch.uint8, device=cfg.device)
# 		audio = torch.from_numpy(artifact["codes"].astype(np.int16))[0, :, :].t().to(dtype=torch.int16, device=cfg.device)

# 		return text, audio

# 	text, audio = load_artifact(f"./data/qnt.{'dac' if cfg.audio_backend == 'dac' else 'enc'}")
# 	batch_size = cfg.hyperparameters.batch_size

# 	text_list = [ text ] * batch_size
# 	proms_list = [ audio[:cfg.dataset.frames_per_second, :] ] * batch_size   ## 1 sec prompt only?
# 	resps_list = [ audio[:cfg.dataset.frames_per_second * 4, :] ] * batch_size ## 3 sec reps as label?

# 	kwargs = {
# 		'n_text_tokens': 256,
# 		'n_audio_tokens': 1024,

# 		'd_model': 1024, # 256, # 1024, # 1536
# 		'n_heads': 16, # 4, # 16, # 24
# 		'n_layers': 12, # 32
# 		'n_experts': 1 if not cfg.model else cfg.model.experts,

# 		'p_dropout': 0.1,

# 		'l_padding': 8 if cfg.optimizations.fp8 else 0,

# 		'config': cfg.model
# 	}

# 	bos_id, space_id, eos_id = cfg.tokenizer.encode( " " )
# 	available_tasks = [] + (["tts-ar"] if "ar" in cfg.model.capabilities else []) + (["tts-nar"] if "len" in cfg.model.capabilities else [])

# 	model = AR_NAR(**kwargs).to(cfg.device)
# 	steps = 500 // batch_size

# 	optimizer = cfg.hyperparameters.optimizer.lower() if cfg.yaml_path is not None else "prodigy"
# 	scheduler = cfg.hyperparameters.scheduler.lower() if cfg.yaml_path is not None else ""
# 	learning_rate = cfg.hyperparameters.learning_rate if cfg.yaml_path is not None else None

# 	params = model.parameters()
# 	if cfg.optimizations.dadaptation:
# 		# do not combine the two
# 		if scheduler == "schedulefree":
# 			scheduler = ""

# 		learning_rate = 1.0
	
# 	if optimizer == "prodigy":
# 		if learning_rate is None:
# 			learning_rate = 1.0

# 		optimizer = ml.Prodigy
# 	elif optimizer == "adagrad":
# 		if learning_rate is None:
# 			learning_rate = 1.0e-2

# 		optimizer = ml.Adagrad
# 	elif optimizer == "adamw":
# 		if learning_rate is None:
# 			learning_rate = 1.0e-4

# 		optimizer = ml.AdamW
# 	elif optimizer == "sdg":
# 		if learning_rate is None:
# 			learning_rate = 1.0e-4

# 		optimizer = ml.SGD
# 	elif optimizer == "apollo":
# 		if learning_rate is None:
# 			learning_rate = 0.01

# 		optimizer = ml.Apollo

# 		"""
# 		target_params = []
# 		target_modules_list = ["attn", "mlp"]
# 		for module_name, module in model.named_modules():
# 			if not (isinstance(module, torch.nn.Linear)):
# 				continue
# 			if not any(target_key in module_name for target_key in target_modules_list):
# 				continue
# 			target_params.append(module.weight)

# 		param_ids = [id(p) for p in target_params]
# 		regular_params = [p for p in model.parameters() if id(p) not in param_ids]
# 		params = [{'params': regular_params}, {'params': target_params, 'rank': 1, 'proj': 'random', 'scale_type': 'tensor', 'scale': 128,'update_proj_gap': 200, 'proj_type': 'std'}]
# 		"""
# 		params = [{'params': params, 'rank': 1, 'proj': 'random', 'scale_type': 'tensor', 'scale': 128,'update_proj_gap': 200, 'proj_type': 'std'}]
# 	else:
# 		raise ValueError(f"Unrecognized optimizer: {optimizer}")

# 	_logger.info(f"Optimizer: {optimizer}\tLearning rate: {learning_rate}")

# 	optimizer = optimizer(params, lr=learning_rate)

# 	if scheduler == "schedulefree":
# 		if isinstance(optimizer, ml.AdamW):
# 			scheduler = ml.schedulefree.AdamWScheduleFree
# 		elif isinstance(optimizer, ml.SGD):
# 			scheduler = ml.schedulefree.SGDScheduleFree
# 		else:
# 			scheduler = None

# 		if scheduler is not None:
# 			_logger.info(f"Scheduler: {scheduler}")
# 			optimizer = scheduler( model.parameters(), lr = learning_rate )

# 	if cfg.optimizations.replace and cfg.optimizations.linear:
# 		model = ml.replace_linear( model )
		
# 	if cfg.optimizations.replace and cfg.optimizations.embedding:
# 		model = ml.replace_embedding( model )

# 	"""
# 	cfg.optimizations.model_offloading = {
# 		"devices": ["cuda:0", "cpu"],
# 	#	"limits": [ 0.9, -1 ],
# 		"assign": [[ f'layers.{i}.' for i in range(0,10) ], [ f'layers.{i}.' for i in range(11,12) ] + [ "model.norm" ]],
# 	#	"limits": [ 256 * (1024 ** 2), -1 ]
# 	}
# 	"""
	
# 	engine = Engine(model=model, optimizer=optimizer)
# 	engines = Engines({"ar+nar": engine})
# 	engines.setup()
	
# 	"""
# 	if cfg.optimizations.model_offloading:
# 		model = ml.offload_model( model, policy=cfg.optimizations.model_offloading )
# 	"""

# 	"""
# 	torch.save( {
# 		'module': model.state_dict()
# 	}, f"./data/{cfg.model.arch_type}.pth" )
# 	"""

# 	_logger.info(f"AR+NAR ({cfg.model.arch_type}, {cfg.audio_backend}) parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 	@torch.no_grad()
# 	def sample_data(t=None):
# 		if isinstance(t, list):
# 			tasks = t
# 			texts = [ text_list[0].to(cfg.device) if task not in text_task else None for i, task in enumerate( tasks ) ]
# 			proms = [ proms_list[0].to(cfg.device) if task not in text_task else [ "stt" ] for i, task in enumerate( tasks ) ]
# 			resps = [ None if task not in text_task else resps_list[0].to(cfg.device) for i, task in enumerate( tasks ) ]

# 			return texts, proms, resps, tasks

# 		texts = []
# 		proms = []
# 		resps = []
# 		tasks = []

# 		## else single batch, sample from possible tasks as well,usually for training, can for inference as well?
# 		for i in range(s): ##batch_size=1 here
# 			task = random.choice(available_tasks) if t is None else t

# 			text = text_list[i].to(cfg.device)
# 			prom = proms_list[i].to(cfg.device)
# 			resp = resps_list[i].to(cfg.device)

# 			# do nothing
# 			if task == "stt":
# 				prom = [ task ]
# 			else:
# 				task = "tts" if random.random() > 0.1 or "len" not in cfg.model.capabilities else "len" ## train len preidictor for 10% of the chance?

# 			texts.append( text )
# 			proms.append( prom )
# 			resps.append( resp )
# 			tasks.append( task )

# 		return texts, proms, resps, tasks

# 	@torch.inference_mode()
# 	def sample( name, steps=500, task=None ):
# 		engine.eval() ## it does not affect model.training=false by default?

# 		text_list, proms_list, resp_list, task_list = sample_data( task )

# 		if task == "tts-nar": # call to forward_ar generate only one reps-level, call to forward_nar, generate for all levels?
# 			len_list = engine( text_list=text_list, proms_list=proms_list, task_list=["len"], max_steps=5, temperature=0.0 ) ## call to forward_ar for len
# 			len_list = [ resp_list[0].shape[0] for l in len_list ] ## what? why don't use the predicted len_list? for training here?
# 			resps_list = engine( text_list=text_list, proms_list=proms_list, len_list=len_list ) ##call to forward-nar automatically once provided with len_list
# 		else: ## tts-ar, normal ar+nar ? 
# 			resps_list = engine( text_list=text_list, proms_list=proms_list, task_list=["tts"], max_duration=steps, temperature=1.0 )  ## call to forward_ar
# 			resps_list = engine( text_list=text_list, proms_list=proms_list, resps_list=resps_list, temperature=0.0 ) # call to forward_nar but not for level 0 without len_list

# 		for i, o in enumerate(resps_list):
# 			_ = decode_to_file(o.to(dtype=torch.int32), f"data/{cfg.model.arch_type}.{cfg.audio_backend}.{i}.{name}.{task}.wav", device=cfg.device)

# 		unload_model()

# 	def train():
# 		engine.train()
# 		t = trange(steps)
# 		for i in t:
# 			texts, proms, resps, tasks = sample_data()

# 			stats = {"step": i}
# 			stats |= engine.traverse(text_list=texts, proms_list=proms, resps_list=resps, task_list=tasks, training=True)
# 			stats |= {"grad_norm": engine.get_global_grad_norm()}

# 			tqdm.write(f"{stats}")

# 		"""
# 		torch.save( {
# 			'module': model.state_dict()
# 		}, f"./data/{cfg.model.arch_type}.pth" )
# 		"""

# 	#sample("init", 5)
# 	train()

# 	"""
# 	if cfg.optimizations.compile:
# 		model = ml.compile_model(model, backend=cfg.optimizations.compile)
# 	"""
	
# 	for task in available_tasks:
# 		sample("final", task=task)

# 	engines.quit()

# if __name__ == "__main__":
# 	example_usage()