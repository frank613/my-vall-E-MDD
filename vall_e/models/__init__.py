import logging

import requests
from tqdm import tqdm
from pathlib import Path

import time
import pdb

_logger = logging.getLogger(__name__)

# to-do: implement automatically downloading model
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / 'data/models'
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "ar+nar-len-llama-8.sft"
DEFAULT_MODEL_URLS = {
	'ar+nar-len-llama-8.sft': 'https://huggingface.co/ecker/vall-e/resolve/main/models/ckpt/ar%2Bnar-len-llama-8/ckpt/fp32.sft',
	'wavlm_large_finetune.pth': 'https://huggingface.co/Dongchao/UniAudio/resolve/main/wavlm_large_finetune.pth',
}

if not DEFAULT_MODEL_PATH.exists() and Path("./data/models/ar+nar-len-llama-8.sft").exists():
	DEFAULT_MODEL_DIR = Path('./data/models')
	DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "ar+nar-len-llama-8.sft"

# kludge, probably better to use HF's model downloader function
# to-do: write to a temp file then copy so downloads can be interrupted
def download_model( save_path=DEFAULT_MODEL_PATH, chunkSize = 1024 ):
	name = save_path.name
	url = DEFAULT_MODEL_URLS[name] if name in DEFAULT_MODEL_URLS else None
	if url is None:
		raise Exception(f'Model requested for download but not defined: {name}')

	if not save_path.parent.exists():
		save_path.parent.mkdir(parents=True, exist_ok=True)

	headers = {}
	# check if modified
	if save_path.exists():
		headers = {"If-Modified-Since": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(save_path.stat().st_mtime))}
	
	r = requests.get(url, headers=headers, stream=True)

	# not modified
	if r.status_code == 304:
		r.close()
		return

	# to-do: validate lengths match
	
	content_length = int(r.headers['Content-Length'] if 'Content-Length' in r.headers else r.headers['content-length'])
	with open(save_path, 'wb') as f:
		bar = tqdm( unit='B', unit_scale=True, unit_divisor=1024, total=content_length, desc=f"Downloading: {name}" )
		for chunk in r.iter_content(chunk_size=chunkSize): 
			if not chunk:
				continue
			bar.update( len(chunk))
			f.write(chunk)
		bar.close()

	r.close()


def get_model(config, training=True, is_mdd=False, **model_kwargs):
	if not is_mdd:	
		from .ar_nar import AR_NAR # import here because reasons
		name = config.name
		model = AR_NAR(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,
			d_model=config.dim,
			n_heads=config.heads,
			n_layers=config.layers,
			n_experts=config.experts,
			
			p_dropout=config.dropout,
			
			l_padding = config.input_alignment,
			
			training = training,
			config = config,
			**model_kwargs
		)
  
	else:
		from .nar_mdd import AR_NAR_MDD # extended for MDD
		name = config.name
		model = AR_NAR_MDD(
			n_text_tokens=config.text_tokens,
			n_audio_tokens=config.audio_tokens,
			d_model=config.dim,
			n_heads=config.heads,
			n_layers=config.layers,
			n_experts=config.experts,
			
			p_dropout=config.dropout,
			
			l_padding = config.input_alignment,
			
			training = training,
			config = config,
			**model_kwargs
		)
	_logger.info(f"{name} ({next(model.parameters()).dtype}): {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

	return model

def get_models(models, training=True, is_mdd=False, **model_kwargs):
	return { model.full_name: get_model(model, training=training, is_mdd=is_mdd, **model_kwargs) for model in models }
