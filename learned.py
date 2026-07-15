#!/usr/bin/env python3
"""
learned.py — learned lookahead prediction + quantization compensation for colibri
Usage:
    python learned.py train    --model /path/to/glm52  # collect data + train
    python learned.py benchmark --model /path/to/glm52 # measure tok/s
"""

import os, sys, time, argparse, torch, torch.nn as nn
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
TOP_K               = 8
LOOKAHEAD           = 3
WINDOW              = 8
N_SEQUENCES         = 50
N_TOKENS_PER_SEQ    = 64
EPOCHS              = 30
BATCH_SIZE          = 512
LR                  = 1e-3
DISK_SPEED_GB_S     = 3.0
EXPERT_SIZE_GB      = 19 / 1024
INT2_EXPERT_SIZE_GB = EXPERT_SIZE_GB / 8

# ============================================================
# QUANTIZATION
# ============================================================
def quantize_to_int2(weight):
    max_val = weight.abs().max()
    scale   = max_val / 1.5
    return torch.clamp(torch.round(weight / scale), -2, 1) * scale

def prequantize_experts(model):
    original = {}
    for idx, layer in enumerate(model.model.layers):
        if not has_experts(layer): continue
        exp = layer.mlp.experts
        original[idx] = (exp.gate_up_proj.data.clone(), exp.down_proj.data.clone())
        exp.gate_up_proj.data = quantize_to_int2(exp.gate_up_proj.data)
        exp.down_proj.data    = quantize_to_int2(exp.down_proj.data)
    print(f"Pre-quantized {len(original)} expert layers to int2")
    return original

def restore_experts(model, original):
    for idx, layer in enumerate(model.model.layers):
        if idx not in original: continue
        layer.mlp.experts.gate_up_proj.data = original[idx][0]
        layer.mlp.experts.down_proj.data    = original[idx][1]

# ============================================================
# HELPERS
# ============================================================
def has_experts(layer):
    return hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts')

def moe_layer_ids(model):
    return [i for i, l in enumerate(model.model.layers) if has_experts(l)]

def capture_forward(model, ids):
    hidden_now  = {}
    routing_now = {}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        if not has_experts(layer): continue
        def make_h(i):
            def hook(module, inp, out):
                hidden_now[i] = inp[0].detach().float()
            return hook
        def make_r(i):
            def hook(module, inp, out):
                logits = out[0] if isinstance(out, tuple) else out
                routing_now[i] = set(
                    torch.topk(logits.detach().float().squeeze(0)[-1:],
                               k=TOP_K, dim=-1).indices[0].tolist()
                )
            return hook
        hooks.append(layer.mlp.register_forward_hook(make_h(idx)))
        hooks.append(layer.mlp.gate.register_forward_hook(make_r(idx)))
    with torch.no_grad():
        out = model(ids)
    for h in hooks: h.remove()
    return out, hidden_now, routing_now

def update_buffer(buf, hidden_now):
    for idx, h in hidden_now.items():
        buf[idx].append(h.squeeze(0)[-1:].cpu())
        if len(buf[idx]) > WINDOW:
            buf[idx].pop(0)

def warmup_buffer(model, ids, moe_ids):
    buf     = {i: [] for i in moe_ids}
    cur_ids = ids.clone()
    for _ in range(WINDOW):
        out, hidden_now, _ = capture_forward(model, cur_ids)
        update_buffer(buf, hidden_now)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cur_ids  = torch.cat([cur_ids, next_tok], dim=1)
    return buf, cur_ids

# ============================================================
# MODELS
# ============================================================
class LinearCompensator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
    def forward(self, x): return self.net(x)

class LookaheadPredictor(nn.Module):
    def __init__(self, dim, n_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, n_experts)
        )
    def forward(self, x): return self.net(x)

# ============================================================
# DATA COLLECTION
# ============================================================
def collect_ar_data(model, tokenizer, text, moe_ids, n_experts):
    all_hidden     = {i: [] for i in moe_ids}
    all_selections = {i: [] for i in moe_ids}
    all_residuals  = {i: [] for i in moe_ids}

    prompts = [s.strip() for s in text.split('.') if len(s.strip()) > 50]
    prompts = prompts[:N_SEQUENCES]

    for seq_idx, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt",
                        max_length=32, truncation=True).input_ids.to("cuda")
        buf = {i: [] for i in moe_ids}

        with torch.no_grad():
            for step in range(N_TOKENS_PER_SEQ):
                # fp16 pass
                hidden_now, fp16_out, routing_now = {}, {}, {}
                hooks = []
                for idx, layer in enumerate(model.model.layers):
                    if not has_experts(layer): continue
                    def make_h(i):
                        def hook(module, inp, out):
                            hidden_now[i] = inp[0].detach().float().squeeze(0)[-1:].cpu()
                            fp16_out[i]   = (out[0] if isinstance(out, tuple) else out).detach().float().squeeze(0)[-1:].cpu()
                        return hook
                    def make_r(i):
                        def hook(module, inp, out):
                            logits = out[0] if isinstance(out, tuple) else out
                            routing_now[i] = torch.topk(
                                logits.detach().float().squeeze(0)[-1:],
                                k=TOP_K, dim=-1
                            ).indices.cpu()
                        return hook
                    hooks.append(layer.mlp.register_forward_hook(make_h(idx)))
                    hooks.append(layer.mlp.gate.register_forward_hook(make_r(idx)))
                out = model(ids)
                for h in hooks: h.remove()

                # int2 pass for residuals
                int2_out = {}
                hooks = []
                for idx, layer in enumerate(model.model.layers):
                    if not has_experts(layer): continue
                    exp     = layer.mlp.experts
                    orig_gu = exp.gate_up_proj.data.clone()
                    orig_d  = exp.down_proj.data.clone()
                    exp.gate_up_proj.data = quantize_to_int2(orig_gu)
                    exp.down_proj.data    = quantize_to_int2(orig_d)
                    def make_i(i, ogu, od, e):
                        def hook(module, inp, out):
                            int2_out[i] = (out[0] if isinstance(out, tuple) else out).detach().float().squeeze(0)[-1:].cpu()
                            e.gate_up_proj.data = ogu
                            e.down_proj.data    = od
                        return hook
                    hooks.append(layer.mlp.register_forward_hook(make_i(idx, orig_gu, orig_d, exp)))
                model(ids)
                for h in hooks: h.remove()

                # update buffer + store
                for i in hidden_now:
                    buf[i].append(hidden_now[i])
                    if len(buf[i]) > WINDOW: buf[i].pop(0)
                    win = torch.cat(buf[i]).mean(0, keepdim=True)
                    all_hidden[i].append(win)
                    if i in routing_now:
                        all_selections[i].append(routing_now[i])
                    if i in fp16_out and i in int2_out:
                        all_residuals[i].append(fp16_out[i] - int2_out[i])

                next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_tok], dim=1)

        if seq_idx % 10 == 0:
            print(f"  sequence {seq_idx+1}/{len(prompts)}")

    return (
        {k: torch.cat(v) for k, v in all_hidden.items() if v},
        {k: torch.cat(v) for k, v in all_selections.items() if v},
        {k: torch.cat(v) for k, v in all_residuals.items() if v}
    )

# ============================================================
# TRAINING
# ============================================================
def train_compensators(hidden, residuals, moe_ids):
    comps = {}
    for i in moe_ids:
        if i not in residuals: continue
        x   = hidden[i].view(-1, hidden[i].shape[-1]).cuda()
        tgt = residuals[i].view(-1, residuals[i].shape[-1]).cuda()
        c   = LinearCompensator(x.shape[-1]).cuda()
        opt = torch.optim.Adam(c.parameters(), lr=1e-4)
        for _ in range(20):
            loss = nn.MSELoss()(c(x), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        comps[i] = c
        print(f"  compensator L{i} done")
    return comps

def train_predictors(hidden, selections, moe_ids, n_experts):
    preds = {}
    for src in moe_ids:
        tgt = src + LOOKAHEAD
        if tgt not in selections: continue
        x   = hidden[src].cuda()
        y   = selections[tgt].cuda()
        lbl = torch.zeros(x.shape[0], n_experts).cuda()
        for i in range(x.shape[0]): lbl[i, y[i]] = 1.0
        p   = LookaheadPredictor(x.shape[-1], n_experts).cuda()
        opt = torch.optim.Adam(p.parameters(), lr=LR)
        crt = nn.BCEWithLogitsLoss()
        for ep in range(EPOCHS):
            perm = torch.randperm(x.shape[0])
            for i in range(0, x.shape[0], BATCH_SIZE):
                idx  = perm[i:i+BATCH_SIZE]
                loss = crt(p(x[idx]), lbl[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        preds[src] = p
        print(f"  predictor L{src} done")
    return preds

# ============================================================
# PATCHING
# ============================================================
def attach_compensators(model, comps):
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        if not has_experts(layer) or idx not in comps: continue
        def make_hook(c):
            def hook(module, inp, out):
                corr = c(inp[0].float()).to(inp[0].dtype)
                if isinstance(out, tuple): return (out[0] + corr,) + out[1:]
                return out + corr
            return hook
        hooks.append(layer.mlp.register_forward_hook(make_hook(comps[idx])))
    print(f"Attached {len(hooks)} compensator hooks")
    return hooks

def predict_experts(predictors, hidden_buf, moe_ids):
    predicted = {}
    for src in moe_ids:
        tgt = src + LOOKAHEAD
        if src not in predictors or not hidden_buf[src]: continue
        h   = torch.cat(hidden_buf[src]).mean(0, keepdim=True).cuda()
        lg  = predictors[src](h)
        predicted[tgt] = set(torch.topk(lg, k=TOP_K, dim=-1).indices[0].tolist())
    return predicted

# ============================================================
# BENCHMARK
# ============================================================
def benchmark(model, tokenizer, predictors, comps, moe_ids, n_tokens=50):
    text     = "The quick brown fox jumps over the lazy dog. " * 20
    ids      = tokenizer(text, return_tensors="pt", max_length=64, truncation=True).input_ids.to("cuda")
    original = prequantize_experts(model)
    hooks    = attach_compensators(model, comps)

    # baseline
    restore_experts(model, original)
    for h in hooks: h.remove()
    times_base = []
    cur = ids.clone()
    with torch.no_grad():
        for _ in range(n_tokens):
            t0 = time.perf_counter()
            time.sleep(len(moe_ids) * TOP_K * (EXPERT_SIZE_GB / DISK_SPEED_GB_S))
            out     = model(cur)
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            cur     = torch.cat([cur, next_tok], dim=1)
            times_base.append(time.perf_counter() - t0)

    # system
    prequantize_experts(model)
    hooks = attach_compensators(model, comps)
    buf, cur = warmup_buffer(model, ids, moe_ids)
    times_sys = []
    with torch.no_grad():
        for _ in range(n_tokens):
            t0 = time.perf_counter()
            out, hidden_now, true_routing = capture_forward(model, cur)
            update_buffer(buf, hidden_now)
            predicted = predict_experts(predictors, buf, moe_ids)
            # disk wait: only misses pay
            wait = sum(
                (TOP_K - len(predicted.get(i, set()) & true_routing.get(i, set()))) * (INT2_EXPERT_SIZE_GB / DISK_SPEED_GB_S)
                for i in moe_ids
            )
            time.sleep(wait)
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=1)
            times_sys.append(time.perf_counter() - t0)

    for h in hooks: h.remove()

    base_toks = n_tokens / sum(times_base)
    sys_toks  = n_tokens / sum(times_sys)
    print(f"\n{'='*50}")
    print(f"RESULTS | {n_tokens} tokens | {DISK_SPEED_GB_S} GB/s NVMe")
    print(f"{'='*50}")
    print(f"Baseline tok/s:   {base_toks:.2f}")
    print(f"System tok/s:     {sys_toks:.2f}")
    print(f"Speedup:          {sys_toks/base_toks:.2f}x")
    print(f"{'='*50}")

# ============================================================
# SAVE / LOAD
# ============================================================
def save(comps, preds, path):
    torch.save({
        'compensators': {k: v.state_dict() for k, v in comps.items()},
        'predictors':   {k: v.state_dict() for k, v in preds.items()},
    }, path)
    print(f"Saved to {path}")

def load(path, moe_ids, hidden_dim, n_experts):
    ckpt  = torch.load(path)
    comps = {}
    for k, sd in ckpt['compensators'].items():
        c = LinearCompensator(hidden_dim).cuda()
        c.load_state_dict(sd)
        comps[int(k)] = c
    preds = {}
    for k, sd in ckpt['predictors'].items():
        p = LookaheadPredictor(hidden_dim, n_experts).cuda()
        p.load_state_dict(sd)
        preds[int(k)] = p
    print(f"Loaded from {path}")
    return comps, preds

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'benchmark'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--weights', default='learned_weights.pt')
    parser.add_argument('--tokens', type=int, default=50)
    args = parser.parse_args()

    # load model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="cuda"
    )
    model.eval()

    moe_ids   = moe_layer_ids(model)
    n_experts = model.model.layers[moe_ids[0]].mlp.experts.num_experts
    hidden_dim = model.config.hidden_size
    print(f"MoE layers: {moe_ids} | experts: {n_experts} | hidden: {hidden_dim}")

    if args.mode == 'train':
        print("\nCollecting autoregressive data...")
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = " ".join([t for t in data["text"] if t.strip()])
        hidden, selections, residuals = collect_ar_data(model, tokenizer, text, moe_ids, n_experts)
        print(f"Collected {list(hidden.values())[0].shape[0]} tokens")

        print("\nTraining compensators...")
        comps = train_compensators(hidden, residuals, moe_ids)

        print("\nTraining predictors...")
        preds = train_predictors(hidden, selections, moe_ids, n_experts)

        save(comps, preds, args.weights)

    elif args.mode == 'benchmark':
        comps, preds = load(args.weights, moe_ids, hidden_dim, n_experts)
        benchmark(model, tokenizer, preds, comps, moe_ids, n_tokens=args.tokens)

if __name__ == '__main__':
    main()