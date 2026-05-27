import torch
import torch.nn as nn
from long_seq import process_long_input
from losses import ATLoss
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import LongformerModel, LongformerTokenizer
from dataclasses import dataclass
from typing import NamedTuple
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

class DocREModel(nn.Module):
    def __init__(self, config, model, dataset='dwie', emb_size=768, block_size=64, num_labels=-1,
                 temperature=1, lambda_sym=0.1, T=2, L=20, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.model = model
        self.hidden_size = config.hidden_size
        self.temperature = temperature
        self.T = T
        self.L = L
        self.n = config.num_labels
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.diff_w = nn.Parameter(torch.Tensor(self.n, self.T, self.L, 2 * self.n + 1))
        nn.init.kaiming_uniform_(self.diff_w.view(self.n, -1), a=np.sqrt(5))
        self.diff_weights = nn.Parameter(torch.Tensor(self.n, self.L, 1))
        nn.init.kaiming_uniform_(self.diff_weights.view(self.n, -1), a=np.sqrt(5))

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=1
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        # Load Longformer model and tokenizer
        self.longformer_model = LongformerModel.from_pretrained(
            '../PLM/longformer-base/',
            attention_window=[256] * config.num_hidden_layers
        )
        self.longformer_model.gradient_checkpointing_enable()
        print('config.num_hidden_layers', config.num_hidden_layers)

        self.alpha_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 2)
        )

        nn.init.constant_(self.alpha_net[-1].weight, 0.0)
        nn.init.constant_(self.alpha_net[-1].bias, 0.0)
        self.alpha_epsilon = 1e-6
        self.lambda_sym = lambda_sym
        print("lambda_sym=",self.lambda_sym)

        self.mamba_dim = config.hidden_size
        self.mamba_encoder = NdMamba2(
            cin=config.hidden_size,
            cout=config.hidden_size,
            mamba_dim=config.hidden_size,
            d_state=64,
            d_conv=4,
            expand=2,
            chunk_size=64
        )

        self.mamba_alpha = nn.Parameter(torch.tensor(0.5))

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)

        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, [0]] = 1

        longformer_output = self.longformer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        longformer_encoded = longformer_output.last_hidden_state

        sequence_output = self.dropout(sequence_output)
        longformer_encoded = self.dropout(longformer_encoded)

        cls_bert = sequence_output[:, 0, :]
        cls_longformer = longformer_encoded[:, 0, :]
        combined_cls = torch.cat([cls_bert, cls_longformer], dim=1)

        raw_alpha = self.alpha_net(combined_cls)

        alpha = torch.nn.functional.softplus(raw_alpha) + self.alpha_epsilon

        dirichlet_dist = torch.distributions.Dirichlet(alpha)

        if self.training:
            weights = dirichlet_dist.rsample()
        else:
            weights = alpha / alpha.sum(dim=1, keepdim=True)

        weights = weights.unsqueeze(-1).unsqueeze(-1)

        sequence_output = weights[:, 0] * sequence_output + weights[:, 1] * longformer_encoded

        sequence_output = sequence_output.transpose(0, 1)
        sequence_output = self.norm(sequence_output + self.dropout(self.transformer_encoder(sequence_output)))
        sequence_output = sequence_output.transpose(0, 1)

        mamba_input = sequence_output.transpose(1, 2)
        mamba_output = self.mamba_encoder(mamba_input)

        sequence_output = sequence_output + self.mamba_alpha * mamba_output.transpose(1, 2)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = torch.einsum("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def add_anti_to_labels(self, labels: torch.Tensor, hts: list) -> torch.Tensor:
        anti_labels = torch.zeros(size=(labels.size(0), labels.size(1) - 1)).to(labels)
        past_entity_pairs = 0
        for hts_one_doc in hts:
            for index, [h, t] in enumerate(hts_one_doc):
                if labels[index + past_entity_pairs, 0] == 1:
                    break
                anti_labels[past_entity_pairs + hts_one_doc.index([t, h])] = labels[index + past_entity_pairs, 1:]

            past_entity_pairs += len(hts_one_doc)
        return torch.cat((labels, anti_labels), dim=1)

    def reasoning_by_soft_rules(self, logits):
        n_e = logits.shape[0]
        eye = torch.eye(n_e).to(self.device)
        input = logits[:, :, :]
        input = torch.cat([input, torch.permute(input, (1, 0, 2)), torch.unsqueeze(eye, dim=-1)], dim=-1)
        all_states = []
        for r in range(self.n):
            cur_states = []
            for t in range(self.T + 1):
                if t == 0:
                    w = self.diff_w[r][t]
                    one_hot = torch.zeros_like(w.detach())
                    if r != 0:
                        one_hot[:, 0] = -1e30
                        one_hot[:, self.n] = -1e30
                    w = torch.softmax(w + one_hot, dim=-1)
                    input_cur = input.view(-1, 2 * self.n + 1)
                    s_tmp = torch.mm(input_cur, torch.permute(w, (1, 0))).view(n_e, -1, self.L)
                    s = s_tmp
                    cur_states.append(s)
                if t >= 1 and t < self.T:
                    w = self.diff_w[r][t]
                    one_hot = torch.zeros_like(w.detach())
                    if r != 0:
                        one_hot[:, 0] = -1e30
                        one_hot[:, self.n] = -1e30
                    w = torch.softmax(w + one_hot, dim=-1)
                    input_cur = torch.permute(input, (0, 2, 1)).reshape(-1, n_e)
                    s_tmp = torch.mm(input_cur, cur_states[t - 1].reshape(n_e, -1))
                    s_tmp = s_tmp.view(n_e, 2 * self.n + 1, -1, self.L)
                    s_tmp = torch.einsum('mrnl,lr->mnl', s_tmp, w)
                    s = s_tmp
                    cur_states.append(s)
                if t == self.T:
                    weight = torch.tanh(self.diff_weights[r])
                    final_state = torch.einsum('mnl,lk->mnk', cur_states[-1], weight).squeeze(dim=-1)
                    all_states.append(final_state)
        output = torch.stack(all_states, dim=-1)
        return output

    def activation(self, x):
        return torch.minimum(torch.maximum(x, torch.zeros_like(x)), torch.ones_like(x))

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                output_for_LogiRE=False,
                ):
        if torch.isnan(input_ids).any():
            input_ids = torch.nan_to_num(input_ids, nan=0)

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))

        hs_rev = torch.tanh(self.head_extractor(torch.cat([ts, rs], dim=1)))
        ts_rev = torch.tanh(self.tail_extractor(torch.cat([hs, rs], dim=1)))

        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits_forward = self.bilinear(bl)

        b1_rev = hs_rev.view(-1, self.emb_size // self.block_size, self.block_size)
        b2_rev = ts_rev.view(-1, self.emb_size // self.block_size, self.block_size)
        bl_rev = (b1_rev.unsqueeze(3) * b2_rev.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits_reverse = self.bilinear(bl_rev)
        logits = logits_forward

        logits_rule_soft = []
        start = 0
        for b in range(len(hts)):
            indices = torch.LongTensor(hts[b]).transpose(1, 0).to(logits)
            n_e = int(np.sqrt(len(hts[b]))) + 1
            end = start + len(hts[b])
            input = torch.softmax(logits[start: end, :], dim=-1)
            matrix = torch.sparse.FloatTensor(indices.long(), input,
                                              torch.Size([n_e, n_e, self.config.num_labels])).to(logits)
            logits_rule = self.reasoning_by_soft_rules(matrix.to_dense())
            logits_rule = logits_rule.view(-1, self.config.num_labels)
            indices = indices[0] * n_e + indices[1]
            logits_rule = logits_rule[indices.long()]
            logits_rule_soft.append(logits_rule)
            start = end
        logits_rule_soft = torch.cat(logits_rule_soft, dim=0) + logits
        if output_for_LogiRE:
            # return logits
            return logits_rule_soft

        output = (self.loss_fnt.get_label(logits_rule_soft, num_labels=self.num_labels),)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)

            loss_cls = self.loss_fnt(logits.float(), labels.float())
            loss_rule = self.loss_fnt(logits_rule_soft.float() / 0.2, labels.clone().float())

            prob_forward = torch.softmax(logits_forward, dim=-1)
            prob_reverse = torch.softmax(logits_reverse, dim=-1)
            loss_sym = F.mse_loss(prob_forward, prob_reverse)

            loss = loss_cls + loss_rule + self.lambda_sym * loss_sym
            loss_dict = {
                'loss_cls': loss_cls.item(),
                'loss_rule': loss_rule.item(),
                'loss_sym': loss_sym.item()
            }
            print(loss_dict)
            output = (loss.to(sequence_output), loss_dict) + output
        return output

@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                    self.pad_vocab_size_multiple
                    - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 vocab_size: int = 50277,
                 pad_vocab_size_multiple: int = 16, ):
        super().__init__()
        args = Mamba2Config(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, vocab_size, pad_vocab_size_multiple)
        self.args = args
        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, ))
        nn.init.constant_(self.dt_bias, -2.0)

        self.A_log = nn.Parameter(torch.empty(args.nheads, ))
        nn.init.uniform_(self.A_log, -8.0, -3.0)

        self.D = nn.Parameter(torch.empty(args.nheads, ))
        nn.init.constant_(self.D, 0.5)

        self.norm = RMSNorm(args.d_inner)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False)

    def forward(self, u: Tensor, h=None):
        if torch.isnan(u).any() or torch.isinf(u).any():
            u = torch.nan_to_num(u, nan=0.0, posinf=1.0, neginf=-1.0)
        if h:
            return self.step(u, h)
        A = -torch.exp(self.A_log.clamp(max=10.0))
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias.clamp(min=-5.0, max=5.0))

        xBC = torch.tanh(xBC)

        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )

        B = torch.tanh(B) * 2.0
        C = torch.tanh(C) * 2.0

        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)

        y, ssm_state = self.stable_ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=x.device,
        )

        y = y + x * self.D.unsqueeze(-1).clamp(min=0.1, max=2.0)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def stable_ssd(self, x, A, B, C, chunk_size, initial_states=None, device = None):
        assert x.shape[1] % chunk_size == 0

        A = A.clamp(max=5.0)
        x, A, B, C = [
            rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
        ]
        A = rearrange(A, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A, dim=-1)
        L = self.stable_segsum(A, device=device)
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        if torch.isnan(Y_diag).any():
            Y_diag = torch.nan_to_num(Y_diag, nan=0.0)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        chunk_A = F.pad(A_cumsum[:, :, :, -1], (1, 0))
        chunk_A = chunk_A.clamp(max=10.0)
        decay_chunk = torch.exp(self.stable_segsum(chunk_A, device=device))

        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        state_decay_out = torch.exp(A_cumsum.clamp(max=10.0))
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        if torch.isnan(Y).any():
            Y = torch.nan_to_num(Y, nan=0.0)

        return Y, final_state

    def step(self, u: Tensor, h: InferenceCache):
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h

    def stable_segsum(self, x: Tensor, device = None) -> Tensor:
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -1e5)
        return torch.exp(x_segsum)

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def silu(x):
    return x * torch.sigmoid(x)

class BaseNdMamba2(nn.Module):
    def __init__(self, cin, cout, mamba_dim, **mamba2_args):
        super().__init__()
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)
        self.mamba2_for = Mamba2(mamba_dim, **mamba2_args)
        self.mamba2_back = Mamba2(mamba_dim, **mamba2_args)
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)

def manual_unflatten(tensor, dim, sizes):
    shape = list(tensor.shape)
    new_shape = shape[:dim] + list(sizes) + shape[dim+1:]
    return tensor.view(new_shape)

class NdMamba2(BaseNdMamba2):
    def __init__(self, cin,  cout, mamba_dim, **mamba2_args):
        super().__init__(cin, cout, mamba_dim, **mamba2_args)


    def forward(self, x):
        size = x.shape[2:]
        x = torch.flatten(x, 2)
        l = x.shape[2]
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))
        x = rearrange(x, 'b c l-> b l c')
        x = self.fc_in(x)
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)
        x = rearrange(x, 'b l c -> b c l')
        x = x[:, :, :l]
        x = manual_unflatten(x, 2, size)
        return x