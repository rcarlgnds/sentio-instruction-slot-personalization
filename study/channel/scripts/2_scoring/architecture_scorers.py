"""Scorer classes for bge-gemma, ctxl-6B, and FollowIR-7B.

Each class follows the inference code in its model's Hugging Face README and keeps the template in
template_info, so every manifest records it. GemmaScorer.get_inputs() is adapted from the
bge-reranker-v2-gemma README under Apache 2.0 (see the note on that method). The ctxl prompt comes from
its README's Transformers example (CC BY-NC-SA 4.0), and the FollowIR template from its README, doubled
bos token included, because that is how the vendor code runs. NOTICE.md lists the licenses.
"""

import os
import subprocess
from pathlib import Path

from model_pins import revision_for

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])


def vram_free_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True)
    return int(out.split()[0])


def wait_for_free_gpu(max_s: int = 180) -> int:
    import os
    import time
    t0 = time.time()
    me = str(os.getpid())
    while time.time() - t0 < max_s:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.split()
        other = [p for p in out if p.strip() and p.strip() != me]
        if not other:
            return int(time.time() - t0)
        time.sleep(2)
    return int(time.time() - t0)


BYTE_DTYPE = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I8": 1, "U8": 1}


def module_sizes_mib(name: str) -> dict:
    import json
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = Path(snapshot_download(name, revision=revision_for(name), allow_patterns=["*.safetensors", "*.index.json"]))
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    sizes, bf16 = {}, {}
    for shard in sorted(set(idx.values())):
        with safe_open(str(d / shard), "pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                n = 1
                for s in sl.get_shape():
                    n *= s
                byte = BYTE_DTYPE.get(sl.get_dtype(), 4)
                if k.startswith("model.layers."):
                    mod = "model.layers." + k.split(".")[2]
                else:
                    mod = k.rsplit(".weight", 1)[0].rsplit(".bias", 1)[0]
                sizes[mod] = sizes.get(mod, 0) + n * (byte + (2 if byte > 2 else 0)) / 2**20
                bf16[mod] = bf16.get(mod, 0) + n * 2 / 2**20
    sizes["__bf16__"] = bf16
    return sizes


def device_map_for(name: str, cap_mib: int) -> dict:
    sizes = module_sizes_mib(name)
    sizes.pop("__bf16__", None)
    layer = sorted((k for k in sizes if k.startswith("model.layers.")),
                   key=lambda k: int(k.split(".")[2]))
    first_keys = [k for k in sizes if k.startswith("model.embed")]
    last_keys = [k for k in sizes if k not in layer and k not in first_keys]
    ordering = first_keys + layer + last_keys
    mapping, taken = {}, 0.0
    for k in ordering:
        if taken + sizes[k] <= cap_mib:
            mapping[k] = 0
            taken += sizes[k]
        else:
            mapping[k] = "cpu"
    return mapping


gap_mib_on_restore = 500


def load_causal_lm(name: str, dtype: str, gpu_cap_mib=None):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"torch_dtype": getattr(torch, dtype)}
    offload = None
    if gpu_cap_mib:
        mapping = device_map_for(name, int(gpu_cap_mib))
        kw["device_map"] = mapping
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision_for(name), **kw)
        cpu = sorted(k for k, v in mapping.items() if v == "cpu")
        bf16 = module_sizes_mib(name)["__bf16__"]
        restored = []
        try:
            from accelerate.hooks import remove_hook_from_module
            for k in cpu:
                if vram_free_mib() - bf16.get(k, 0) < gap_mib_on_restore + 400:
                    break
                sub = model.get_submodule(k)
                remove_hook_from_module(sub)
                sub.to("cuda")
                restored.append(k)
            if restored and len(restored) == len(cpu):
                remove_hook_from_module(model, recurse=True)
                model.to("cuda")
        except Exception as e:
            print(f"restore to GPU stopped: {type(e).__name__}: {e}", flush=True)
        offload = {"cap_gpu_mib": int(gpu_cap_mib),
                   "n_modul_gpu": sum(1 for v in mapping.values() if v == 0),
                   "n_modul_cpu_saat_muat": len(cpu), "modul_cpu_saat_muat": cpu,
                   "dipulihkan_ke_gpu_setelah_cast": restored,
                   "n_modul_cpu_saat_jalan": len(cpu) - len(restored),
                   "cara": "explicit device map from the safetensors header; the modules that fit move back to the "
                           "GPU after the bf16 cast"}
    else:
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision_for(name), **kw).cuda()
    model.eval()
    return model, offload


def _truncate(tok, text: str, budget: int):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text, False
    return tok.decode(ids[:budget], skip_special_tokens=True), True


class GemmaScorer:
    PROMPT_DEFAULT = ("Given a query A and a passage B, determine whether the "
                      "passage contains an answer to the query by providing a "
                      "prediction of either 'Yes' or 'No'.")

    def __init__(self, name: str, dtype: str, gpu_cap_mib=None):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision_for(name))
        self.tok.padding_side = "left"
        self.model, self.offload = load_causal_lm(name, dtype, gpu_cap_mib)
        self.yes_loc = self.tok("Yes", add_special_tokens=False)["input_ids"][0]
        self.prompt = self.PROMPT_DEFAULT
        self.n_dokumen_terpotong = 0
        self.template_info = {
            "sumber": "README of BAAI/bge-reranker-v2-gemma, get_inputs function adapted (see NOTICE.md)",
            "format": "<bos>A: {query}\\nB: {passage}\\n{prompt}",
            "skor": "raw logit of the token 'Yes' at the last position, no softmax",
            "slot_instruksi": "the prompt argument; the profile is appended after the vendor default prompt",
            "prompt_bawaan_verbatim": self.PROMPT_DEFAULT,
            "padding_side": "left (forced; the README does not set it)",
            "yes_token_id": int(self.yes_loc),
        }

    def set_instruction(self, inst):
        self.prompt = self.PROMPT_DEFAULT if inst is None else f"{self.PROMPT_DEFAULT} {inst}"

    def set_full_prompt(self, prompt):
        self.prompt = prompt

    def get_inputs(self, pairs, prompt, max_length):
        """Adapted from get_inputs() in the BAAI/bge-reranker-v2-gemma README
        (Apache 2.0, see NOTICE.md). Here the tokenizer is bound to self, the
        prompt comes from the caller, and a counter of truncated documents is
        added."""
        tokenizer = self.tok
        sep = "\n"
        prompt_inputs = tokenizer(prompt, return_tensors=None,
                                  add_special_tokens=False)["input_ids"]
        sep_inputs = tokenizer(sep, return_tensors=None,
                               add_special_tokens=False)["input_ids"]
        inputs = []
        for query, passage in pairs:
            query_inputs = tokenizer(f"A: {query}", return_tensors=None,
                                     add_special_tokens=False,
                                     max_length=max_length * 3 // 4,
                                     truncation=True)
            passage_inputs = tokenizer(f"B: {passage}", return_tensors=None,
                                       add_special_tokens=False,
                                       max_length=max_length, truncation=True)
            if len(query_inputs["input_ids"]) + len(sep_inputs) + len(
                    passage_inputs["input_ids"]) + 1 > max_length:
                self.n_dokumen_terpotong += 1
            item = tokenizer.prepare_for_model(
                [tokenizer.bos_token_id] + query_inputs["input_ids"],
                sep_inputs + passage_inputs["input_ids"],
                truncation="only_second", max_length=max_length, padding=False,
                return_attention_mask=False, return_token_type_ids=False,
                add_special_tokens=False)
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)
        return tokenizer.pad(inputs, padding=True,
                             max_length=max_length + len(sep_inputs) + len(prompt_inputs),
                             pad_to_multiple_of=8, return_tensors="pt")

    def score(self, query_text: str, docs: list, batch: int, max_len: int) -> list:
        torch = self.torch
        output = []
        with torch.inference_mode():
            for i in range(0, len(docs), batch):
                pairs = [(query_text, d) for d in docs[i:i + batch]]
                enc = self.get_inputs(pairs, self.prompt, max_len)
                enc = {k: v.cuda() for k, v in enc.items()}
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                output.extend(logits[:, self.yes_loc].float().tolist())
        return output


class CtxlScorer:
    FRAME = ("Check whether a given document contains information helpful "
                "to answer the query.\n<Document> {doc}\n<Query> {query}{inst} ??")

    def __init__(self, name: str, dtype: str, gpu_cap_mib=None):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision_for(name), use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model, self.offload = load_causal_lm(name, dtype, gpu_cap_mib)
        self.inst = None
        self.n_dokumen_terpotong = 0
        self.template_info = {
            "sumber": "README of ContextualAI ctxl-rerank-v2, the Transformers path infer_w_hf, with each document cut to the token budget",
            "format": self.FRAME,
            "skor": "logit of token id 0 at the last position, cast to bfloat16 and then to float",
            "true_token_id": 0, "false_token_id": None,
            "slot_instruksi": "the instruction is appended after the query with one space; empty means the vendor default",
            "bos": "one, added by the tokenizer (add_bos_token true); the string holds no <s>",
        }

    def set_instruction(self, inst):
        self.inst = inst

    def prompt(self, query_text: str, doc: str) -> str:
        return self.FRAME.format(doc=doc, query=query_text,
                                    inst=(f" {self.inst}" if self.inst else ""))

    def score(self, query_text: str, docs: list, batch: int, max_len: int) -> list:
        torch = self.torch
        overhead = len(self.tok(self.prompt(query_text, ""))["input_ids"])
        budget = max(16, max_len - overhead)
        text = []
        for d in docs:
            t, cut = _truncate(self.tok, d, budget)
            self.n_dokumen_terpotong += cut
            text.append(self.prompt(query_text, t))
        output = []
        with torch.inference_mode():
            for i in range(0, len(text), batch):
                enc = self.tok(text[i:i + batch], return_tensors="pt", padding=True,
                               truncation=True, max_length=max_len)
                enc = {k: v.cuda() for k, v in enc.items()}
                logits = self.model(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"],
                                    logits_to_keep=1).logits[:, -1, :]
                output.extend(logits[:, 0].to(torch.bfloat16).float().tolist())
        return output


class FollowIRScorer:
    TEMPLATE = ("<s> [INST] You are an expert Google searcher, whose job is to "
                "determine if the following document is relevant to the query "
                "(true/false). Answer using only one word, one of those two "
                "choices.\n\nQuery: {query}\nDocument: {text}\nRelevant (only "
                "output one word, either \"true\" or \"false\"): [/INST] ")

    def __init__(self, name: str, dtype: str, gpu_cap_mib=None):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision_for(name), padding_side="left")
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model, self.offload = load_causal_lm(name, dtype, gpu_cap_mib)
        v = self.tok.get_vocab()
        self.id_false, self.id_true = v["false"], v["true"]
        self.inst = None
        self.n_dokumen_terpotong = 0
        self.template_info = {
            "sumber": "README of jhu-clsp/FollowIR-7B, template and scoring copied verbatim",
            "format": self.TEMPLATE,
            "skor": "softmax(logit false, logit true)[true] at the last position",
            "true_token_id": int(self.id_true), "false_token_id": int(self.id_false),
            "slot_instruksi": "appended to the query text ('Query: {query} {instruksi}'), the FollowIR convention",
            "bos": "double (the template has <s> and the tokenizer adds bos), as in the README",
        }

    def set_instruction(self, inst):
        self.inst = inst

    def prompt(self, query_text: str, doc: str) -> str:
        q = f"{query_text} {self.inst}" if self.inst else query_text
        return self.TEMPLATE.format(query=q, text=doc)

    def score(self, query_text: str, docs: list, batch: int, max_len: int) -> list:
        torch = self.torch
        overhead = len(self.tok(self.prompt(query_text, ""))["input_ids"])
        budget = max(16, max_len - overhead)
        text = []
        for d in docs:
            t, cut = _truncate(self.tok, d, budget)
            self.n_dokumen_terpotong += cut
            text.append(self.prompt(query_text, t))
        output = []
        with torch.inference_mode():
            for i in range(0, len(text), batch):
                enc = self.tok(text[i:i + batch], padding=True, truncation=True,
                               max_length=max_len, return_tensors="pt",
                               pad_to_multiple_of=None)
                enc = {k: v.cuda() for k, v in enc.items()}
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                duo = torch.stack([logits[:, self.id_false], logits[:, self.id_true]], dim=1)
                output.extend(torch.nn.functional.log_softmax(
                    duo.float(), dim=1)[:, 1].exp().tolist())
        return output


NEW_FAMILY = {"gemma": GemmaScorer, "ctxl": CtxlScorer, "followir": FollowIRScorer}
MAX_BATCH = {"gemma": 8, "ctxl": 8, "followir": 2, "zerank2": 4}
