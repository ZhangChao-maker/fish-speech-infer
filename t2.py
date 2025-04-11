import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.conversation import (
    CODEBOOK_PAD_TOKEN_ID,
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from fish_speech.models.text2semantic.llama import BaseModelArgs
from fish_speech.text import clean_text, split_text
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=0, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync_agent(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: torch.Tensor = 1.0,
    top_p: torch.Tensor = 1.0,
    repetition_penalty: torch.Tensor = 1.0,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def sample_agent(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs_agent(
        logits=logits[:, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync_agent(probs)
    return idx_next, probs


def decode_one_token_ar_agent(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # print(x, input_pos)
    x = model.forward_generate(x, input_pos)
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    sampling_kwargs_main = sampling_kwargs.copy()
    sampling_kwargs_main["temperature"] = 0.1
    sampling_kwargs_main["top_p"] = 0.1
    sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample_agent(
            logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample_agent(
            logits,
            previous_tokens=(
                previous_tokens[:, codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~torch.isin(codebooks[:, :1, :], semantic_ids_tensor),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks


def decode_one_token_naive_agent(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    codebooks = [
        sample(
            x.token_logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample_agent(
                x.codebook_logits[:, :, i],
                previous_tokens=(
                    previous_tokens[:, i + 1] if previous_tokens is not None else None
                ),
                **sampling_kwargs,
            )[0]
        )

    codebooks = torch.stack(codebooks, dim=1)
    semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    codebooks[:, 1:, :] = torch.masked_fill(
        codebooks[:, 1:, :],
        ~torch.isin(codebooks[:, :1, :], semantic_ids_tensor),
        CODEBOOK_PAD_TOKEN_ID,
    )

    return codebooks


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    semantic_ids: list,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    sampling_kwargs_main = sampling_kwargs.copy()
    # sampling_kwargs_main["temperature"] = 0.1
    # sampling_kwargs_main["top_p"] = 0.1
    # sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample(
            x.logits,
            previous_tokens=(
                previous_tokens[0] if previous_tokens is not None else None
            ),  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    hidden_states = x.hidden_states

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=0)
    # semantic_ids_tensor = torch.tensor(semantic_ids, device=codebooks.device)
    # codebooks[1:, :] = torch.masked_fill(
    #     codebooks[1:, :], ~torch.isin(codebooks[:1, :], semantic_ids_tensor), CODEBOOK_PAD_TOKEN_ID
    # )

    # print(codebooks)
    return codebooks


def decode_one_token_naive(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    sampling_kwargs_main = sampling_kwargs.copy()
    sampling_kwargs_main["temperature"] = 0.1
    sampling_kwargs_main["top_p"] = 0.1
    sampling_kwargs_main["repetition_penalty"] = 1.0

    codebooks = [
        sample(
            x.logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs_main,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample(
                x.codebook_logits[:, :, i],
                previous_tokens=(
                    previous_tokens[i + 1] if previous_tokens is not None else None
                ),
                **sampling_kwargs,
            )[0]
        )

    return torch.stack(codebooks, dim=0)


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with (
            torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
            if torch.cuda.is_available()
            else nullcontext()
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: NaiveTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    # semantic_id = model.tokenizer.convert_tokens_to_ids("<|semantic|>")
    semantic_ids = [
        model.tokenizer.get_token_id(f"<|semantic:{i}|>") for i in range(1024)
    ]

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    codebook_dim = 1 + model.config.num_codebooks
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (
        decode_one_token_naive
        if isinstance(model, NaiveTransformer)
        else decode_one_token_ar
    )

    next_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )
    seq[:, T : T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        decode_one_token=decode_one_token,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    )
    # x = torch.cat(generated_tokens, dim=1)
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    return seq


def decode_n_tokens_agent(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    batch_size = cur_token.size(0)
    previous_tokens = torch.zeros(
        (batch_size, model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=cur_token.device)
    finished = finished | (cur_token[:, 0, -1] == im_end_id)
    start_time = time.time()

    for i in tqdm(range(num_new_tokens), desc="Decoding: ", total=num_new_tokens):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :, :win_size]
        else:
            window = previous_tokens[:, :, i - win_size : i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                semantic_ids=semantic_ids,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(batch_size, model.config.num_codebooks + 1, -1)
        previous_tokens[:, :, i : i + 1] = next_token.view(
            batch_size, model.config.num_codebooks + 1, -1
        )

        yield cur_token.cpu()

        finished = finished | (cur_token[:, 0, -1] == im_end_id)
        if finished.all() or (
            0 < early_stop_threshold < 1
            and finished.sum() >= round(batch_size * early_stop_threshold)
        ):
            break

    total_time = time.time() - start_time
    generated_tokens = i + 1
    tokens_per_second = (generated_tokens / total_time) * batch_size
    logger.info(
        f"Decoded {generated_tokens} x {batch_size} tokens in {total_time:.2f}s ({tokens_per_second:.2f} tokens/s)"
    )


@torch.no_grad()
@torch.inference_mode()
def generate_agent(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    semantic_ids: list,
    im_end_id: int = 4,
    decode_one_token=decode_one_token_naive_agent,
    num_samples: int = 1,
    early_stop_threshold: float = 0.6,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)

    # Use non-accelerated version for now, to avoid compilation overhead
    prefill_decode = (
        decode_one_token_naive_agent
        if isinstance(model, NaiveTransformer)
        else decode_one_token_ar_agent
    )
    next_token = prefill_decode(
        model,
        prompt,
        input_pos,
        semantic_ids=semantic_ids,
        **sampling_kwargs,
    ).view(num_samples, codebook_dim, -1)
    yield next_token.cpu()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    yield from decode_n_tokens_agent(
        model,
        next_token,
        input_pos,
        max_new_tokens - 1,
        im_end_id=im_end_id,
        semantic_ids=semantic_ids,
        decode_one_token=decode_one_token,
        early_stop_threshold=early_stop_threshold,
        **sampling_kwargs,
    )


def encode_tokens(
    tokenizer,
    string,
    device="cuda",
    prompt_tokens=None,
    num_codebooks=4,
):
    string = clean_text(string)

    messages = []
    messages.append(
        Message(
            role="user",
            parts=[TextPart(text=string)],
            cal_loss=False,
        )
    )

    if prompt_tokens is not None:
        if prompt_tokens.ndim == 3:
            assert (
                prompt_tokens.shape[0] == 1
            ), "3D prompt tokens should have shape (1, num_codebooks, seq_len)"
            prompt_tokens = prompt_tokens[0]

        assert prompt_tokens.ndim == 2, "Prompt tokens should be 2D tensor"

        if prompt_tokens.shape[0] > num_codebooks:
            logger.warning(
                f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
            )
            prompt_tokens = prompt_tokens[:num_codebooks]

        vq_part = VQPart(codes=prompt_tokens.to(device))

        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>"), vq_part],
                cal_loss=False,
            )
        )
    else:
        messages.append(
            Message(
                role="assistant",
                parts=[TextPart(text="<|voice|>")],
                cal_loss=False,
                add_im_end=False,
            )
        )

    conversation = Conversation(messages=messages)
    # conversation.visualize(tokenizer)
    encoded = conversation.encode_for_inference(
        tokenizer=tokenizer,
        num_codebooks=num_codebooks,
    )

    return encoded.to(device)


def load_model(checkpoint_path, device, precision, compile=False, is_agent=False):
    model: Union[NaiveTransformer, DualARTransformer] = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = (
            decode_one_token_ar_agent if is_agent else decode_one_token_ar
        )
        logger.info("Using DualARTransformer")
    else:
        decode_one_token = (
            decode_one_token_naive_agent if is_agent else decode_one_token_naive
        )
        logger.info("Using NaiveTransformer")

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 2,
    max_new_tokens: int = 0,
    top_p: int = 0.7,
    repetition_penalty: float = 1.5,
    temperature: float = 0.7,
    compile: bool = False,
    iterative_prompt: bool = True,
    max_length: int = 2048,
    chunk_length: int = 150,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    im_end_id = tokenizer.get_token_id("<|im_end|>")

    encoded = []
    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    encoded_prompts = [
        Conversation(
            messages=[
                Message(
                    role="system",
                    parts=[TextPart(text="Speak out the provided text.")],
                    cal_loss=False,
                )
            ]
        )
        .encode_for_inference(
            tokenizer=tokenizer,
            num_codebooks=model.config.num_codebooks,
        )
        .to(device)
    ]

    if use_prompt:
        for idx, (t, c) in enumerate(zip(prompt_text, prompt_tokens)):
            encoded_prompts.append(
                encode_tokens(
                    tokenizer,
                    string=t,
                    device=device,
                    prompt_tokens=c,
                    num_codebooks=model.config.num_codebooks,
                )
            )

    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                device=device,
                num_codebooks=model.config.num_codebooks,
            )
        )
        logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        seg_idx = 0

        while seg_idx < len(encoded):
            logger.info(
                f"Generating sentence {seg_idx + 1}/{len(encoded)} of sample {sample_idx + 1}/{num_samples}"
            )

            seg = encoded[seg_idx]
            global_encoded.append(seg)

            lengths = reversed([seg.size(1) for seg in global_encoded])

            # Pick last 2000 tokens
            count = 0
            for i, length in enumerate(lengths):
                count += length
                if count + length > max_length - 1024 - sum(
                    t.shape[1] for t in encoded_prompts
                ):
                    break

            if i != 0 and i % 2 == 0:
                i -= 1

            # Rotate the list, always make sure first segment is included to avoid drift
            if i < len(global_encoded) - 2:
                partial_encoded = global_encoded[:2] + global_encoded[-i:]
            else:
                partial_encoded = global_encoded

            if use_prompt:
                partial_encoded = encoded_prompts + partial_encoded

            cat_encoded = torch.cat(partial_encoded, dim=1)
            prompt_length = cat_encoded.size(1)

            t0 = time.perf_counter()
            y = generate(
                model=model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if sample_idx == 0 and seg_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t = time.perf_counter() - t0

            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t
            logger.info(
                f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            if torch.cuda.is_available():
                logger.info(
                    f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
                )

            # Put the generated tokens
            # since there is <im_end>, we remove last token
            codes = y[1:, prompt_length + 1 :].clone()
            assert (codes >= 0).all(), f"Negative code found"

            decoded = y[:, prompt_length:].clone()
            # But for global encoding, we should keep the <im_end> token

            global_encoded.append(decoded)
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            yield GenerateResponse(action="sample", codes=codes, text=texts[seg_idx])
            seg_idx += 1

        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = load_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


def launch_thread_safe_queue_agent(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = BaseModelArgs.from_pretrained(checkpoint_path)

    def worker():
        model, decode_one_token = load_model(
            checkpoint_path, device, precision, compile=compile, is_agent=True
        )

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for token in generate_agent(
                    model=model,
                    decode_one_token=decode_one_token,
                    **kwargs,
                ):
                    response_queue.put(token)

                response_queue.put("stop")
            except Exception as e:
                import traceback

                logger.exception(f"Error in worker: {traceback.format_exc()}")
                response_queue.put("error")

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue, tokenizer, config


import os
import time
import warnings
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from typing import Optional, List, Tuple
from loguru import logger
from hydra.utils import instantiate
import hydra

warnings.filterwarnings("ignore", category=FutureWarning)

class TTSGenerator:
    def __init__(
        self,
        # TTS模型参数
        tts_checkpoint: Path = Path("checkpoints/fish-speech-1.5"),
        device: str = "cuda",
        compile_model: bool = False,
        seed: int = 42,
        half_precision: bool = False,
        chunk_size: int = 100,
        output_dir: Path = Path("output"),
        max_new_tokens: int = 1024,
        sampling_params: dict = {
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "temperature": 0.7,
        },
        
        # 声码器参数
        vocoder_config_path: Path = Path("../configs/firefly_gan_vq.yaml"),
        vocoder_checkpoint: Path = Path("checkpoints/firefly-gan-vq.pth"),
        sample_rate: int = 44100
    ):
        """全功能语音合成生成器
        
        Args:
            tts_checkpoint: TTS模型检查点路径
            vocoder_config_path: 声码器配置文件路径
            vocoder_checkpoint: 声码器模型检查点路径
            sample_rate: 目标采样率
        """
        self.device = device
        self.output_dir = output_dir
        self.sampling_params = sampling_params
        
        # 初始化目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化模型
        self._init_tts_model(
            tts_checkpoint=tts_checkpoint,
            compile_model=compile_model,
            half_precision=half_precision,
            seed=seed,
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
        )
        
        # 初始化声码器
        self._init_vocoder(
            vocoder_config_path=vocoder_config_path,
            vocoder_checkpoint=vocoder_checkpoint,
            sample_rate=sample_rate,
        )

    def _init_tts_model(
        self,
        tts_checkpoint: Path,
        compile_model: bool,
        half_precision: bool,
        seed: int,
        chunk_size: int,
        max_new_tokens: int,
    ):
        """初始化文本到语义模型"""
        logger.info("Initializing TTS model...")
        
        # 设置随机种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 加载模型
        precision = torch.half if half_precision else torch.bfloat16
        self.model, self.decode_fn = load_model(
            tts_checkpoint, self.device, precision, compile=compile_model
        )
        
        # 初始化缓存
        with torch.device(self.device):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=next(self.model.parameters()).dtype,
            )
            
        logger.success(f"TTS模型加载完成 | 最大生成长度: {max_new_tokens} tokens")

    def _init_vocoder(
        self,
        vocoder_config_path: Path,
        vocoder_checkpoint: Path,
        sample_rate: int
    ):
        """初始化声码器模型"""
        logger.info("Initializing vocoder...")
        
        # 初始化 Hydra
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize(version_base="1.3", config_path="./configs"):
            cfg = hydra.compose(config_name="firefly_gan_vq")
        # with hydra.initialize(version_base="1.3", config_path=vocoder_config_path):
        #     cfg = hydra.compose(
        #         config_name=vocoder_config_path.stem,
        #         overrides=[f"sample_rate={sample_rate}"]
        #     )
        
        # 加载模型
        self.vocoder = instantiate(cfg)
        state_dict = torch.load(vocoder_checkpoint, map_location=self.device)
        self.vocoder.load_state_dict(
            state_dict.get("state_dict", state_dict), 
            strict=False
        )
        self.vocoder.eval().to(self.device)
        logger.success("声码器加载完成")

    def generate_audio(
        self,
        text: str,
        prompt_text: list[str],
        prompt_tokens: list[str],
        output_name: str = "output",
        save_codes: bool = False,
        play_audio: bool = False
    ) -> Tuple[str, str]:
        """端到端语音合成
        
        Args:
            text: 目标文本内容
            prompt_text: 参考文本列表
            prompt_tokens: 参考代码文件路径列表
            output_name: 输出文件基础名称
            save_codes: 是否保存中间语义代码
            play_audio: 是否自动播放
        
        Returns:
            (音频路径, 代码路径)
        """
        # 生成语义代码
        codes = self._generate_codes(text, prompt_text, prompt_tokens)
        
        # 转换代码为音频
        audio_path = self._codes_to_audio(codes, output_name)
        
        # 保存代码文件
        code_path = None
        if save_codes:
            code_path = self._save_codes(codes, output_name)
        
        # 自动播放
        if play_audio:
            self._play_audio(audio_path)
        
        return (audio_path, code_path)

    def _generate_codes(
        self,
        text: str,
        prompt_text: List[str],
        prompt_tokens:str
    ) -> np.ndarray:
        """生成语义编码"""
        logger.debug("开始生成语义编码...")
        
        # 验证输入
        if prompt_text and prompt_tokens:
            if len(prompt_text) != len(prompt_tokens):
                raise ValueError(
                    f"参考文本数量({len(prompt_text)})与代码文件数量({len(prompt_tokens)})不匹配"
                )
        
        # 预处理prompt tokens
        if prompt_tokens is not None:
            prompt_tokens = [torch.from_numpy(np.load(p)).to(self.device) for p in prompt_tokens]
        
        # 生成代码
        # print(prompt_tokens)
        t1=time.time()
        generator = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_fn,
            text=text,
            num_samples=1,
            max_new_tokens=self.sampling_params.get("max_new_tokens", 1024),
            chunk_length=self.sampling_params.get("chunk_size", 100),
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            **self.sampling_params
        )
        t2=time.time()
        print("生成时间:",t2-t1)
        
        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
                logger.info(f"Sampled text: {response.text}")
            elif response.action == "next":
                continue  # 继续生成下一个样本
        print("完成时间", time.time()-t2)



        
        # for response in generator:
        #     if response.action == "next" and codes:
        #         break  # 只取第一个样本
        #     if response.action == "sample":
        #         codes.append(response.codes)
        
        return torch.cat(codes, dim=1).cpu().numpy()

    def _codes_to_audio(
        self,
        codes: np.ndarray,
        output_name: str
    ) -> str:
        """语义编码转音频波形"""
        logger.debug("转换语义编码到音频波形...")
        
        # 转换为张量并生成
        with torch.no_grad():
            indices = torch.from_numpy(codes).to(self.device).long()
            fake_audios, _ = self.vocoder.decode(
                indices=indices[None],
                feature_lengths=torch.tensor([indices.shape[1]], device=self.device)
            )
            waveform = fake_audios[0, 0].float().cpu().numpy()
        
        # 保存文件
        output_path = self.output_dir / f"{output_name}.wav"
        sf.write(output_path, waveform, self.vocoder.spec_transform.sample_rate)
        logger.success(f"音频文件已保存: {output_path}")
        
        return str(output_path)

    def _save_codes(
        self,
        codes: np.ndarray,
        output_name: str
    ) -> str:
        """保存语义编码"""
        output_path = self.output_dir / f"{output_name}_codes.npy"
        np.save(output_path, codes)
        logger.info(f"语义代码已保存: {output_path}")
        return str(output_path)

    def _play_audio(self, file_path: str):
        """自动播放音频"""
        try:
            from pydub import AudioSegment
            from pydub.playback import play
            
            audio = AudioSegment.from_wav(file_path)
            play(audio)
        except ImportError:
            logger.warning("播放需要安装pydub: pip install pydub")
        except Exception as e:
            logger.error(f"播放失败: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 初始化生成器
    tts = TTSGenerator(
        tts_checkpoint="../checkpoints/fish-speech-1.5",
        vocoder_checkpoint="../checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        sample_rate=44100,
        # compile_model=True
    )
    
    # 生成音频
    audio_path, _ = tts.generate_audio(
        text="厚,你真的很烦内,干嘛一直偷看人家啦！",
        prompt_tokens=["fake.npy"],
        output_name="demo",
        play_audio=True,
        save_codes=False,
        prompt_text=["今天天气真是太好了，阳光灿烂，心情超级棒！但是，朋友最近的感情问题也让我心痛不已，好像世界末日一样，真的好为她难过哦！"]
    )
    audio_path, _ = tts.generate_audio(
        text="厚,你真的很烦内,干嘛一直偷看人家啦！",
        prompt_tokens=["fake.npy"],
        output_name="demo",
        play_audio=True,
        save_codes=False,
        prompt_text=["今天天气真是太好了，阳光灿烂，心情超级棒！但是，朋友最近的感情问题也让我心痛不已，好像世界末日一样，真的好为她难过哦！"]
    )
    audio_path, _ = tts.generate_audio(
        text="厚,你真的很烦内,干嘛一直偷看人家啦！",
        prompt_tokens=["fake.npy"],
        output_name="demo",
        play_audio=True,
        save_codes=False,
        prompt_text=["今天天气真是太好了，阳光灿烂，心情超级棒！但是，朋友最近的感情问题也让我心痛不已，好像世界末日一样，真的好为她难过哦！"]
    )
    
    
    '''
    测试时间：11-12秒
    '''