import torch
import torch.nn as nn


class SPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, use_embed_for_s_prompt=False):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_embed_for_s_prompt = use_embed_for_s_prompt


        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers * 2, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(
                            prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                # if self.use_embed_for_s_prompt:
                #     prompt_pool_shape = (self.pool_size,self.length,embed_dim)
                #     self.prompt = torch.embedding()
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            similarity = similarity.t()  # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
            out['similarity'] = similarity

            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id,
                                           torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()),
                                                      device=prompt_id.device)])
                    id_counts = torch.cat(
                        [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                major_prompt_id = prompt_id[major_idx]  # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k

            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = self.prompt[:, idx]
                # batched_prompt_raw = self.prompt[:, :, idx]# num_layers, B, top_k, length, C
                num_layers_dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                # batched_prompt = batched_prompt_raw.reshape(
                #     num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                # )
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers_dual, batch_size, num_heads, top_k * length, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:, idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx]  # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed_mean.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['batched_prompt'] = batched_prompt

        return out


class SPrompt_mul(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, s_pool_size=None, s_top_k=None,t_pool_size=None,t_top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, use_embed_for_s_prompt=False):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_embed_for_s_prompt = use_embed_for_s_prompt


        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    s_prompt_pool_shape = (self.num_layers*2, self.s_pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.s_prompt = nn.Parameter(torch.randn(
                            s_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.s_prompt, -1, 1)

                    t_prompt_pool_shape = (self.num_layers*2, self.t_pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.t_prompt = nn.Parameter(torch.randn(
                            t_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.t_prompt, -1, 1)

            else:
                # if self.use_embed_for_s_prompt:
                #     prompt_pool_shape = (self.pool_size,self.length,embed_dim)
                #     self.prompt = torch.embedding()
                s_prompt_pool_shape = (self.num_layers, self.s_pool_size, self.length, embed_dim)
                t_prompt_pool_shape = (self.num_layers,self.t_pool_size,self.length,embed_dim)
                if prompt_init == 'zero':
                    self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                    self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.s_prompt = nn.Parameter(torch.randn(s_prompt_pool_shape))
                    self.t_prompt = nn.Parameter(torch.randn(s_prompt_pool_shape))
                    nn.init.uniform_(self.s_prompt, -1, 1)
                    nn.init.uniform_(self.t_prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            s_key_shape = (s_pool_size, embed_dim)
            t_key_shape = (t_pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.s_prompt_key = nn.Parameter(torch.zeros(s_key_shape))
                self.t_prompt_key = nn.Parameter(torch.zeros(s_key_shape))
            elif prompt_key_init == 'uniform':
                self.s_prompt_key = nn.Parameter(torch.randn(s_key_shape))
                nn.init.uniform_(self.s_prompt_key, -1, 1)
                self.t_prompt_key = nn.Parameter(torch.randn(t_key_shape))
                nn.init.uniform_(self.t_prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            s_prompt_mean = torch.mean(self.s_prompt, dim=[0, 2])
            self.s_prompt_key = s_prompt_mean
            t_prompt_mean = torch.mean(self.t_prompt, dim=[0, 2])
            self.t_prompt_key = t_prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, s_prompt_mask=None,t_prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            s_prompt_key_norm = self.l2_normalize(self.s_prompt_key, dim=-1)  # Pool_size, C
            t_prompt_key_norm = self.l2_normalize(self.t_prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            s_similarity = torch.matmul(s_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            s_similarity = s_similarity.t()  # B, pool_size
            t_similarity = torch.matmul(t_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            t_similarity = t_similarity.t()  # B, pool_size

            (s_similarity_top_k, s_idx) = torch.topk(s_similarity, k=self.s_top_k, dim=1)  # B, top_k
            out['s_similarity'] = s_similarity
            (t_similarity_top_k, t_idx) = torch.topk(t_similarity, k=self.t_top_k, dim=1)  # B, top_k
            out['t_similarity'] = t_similarity

            if self.batchwise_prompt:
                s_prompt_id, s_id_counts = torch.unique(s_idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if s_prompt_id.shape[0] < self.s_pool_size:
                    s_prompt_id = torch.cat([s_prompt_id,
                                           torch.full((self.s_pool_size - s_prompt_id.shape[0],), torch.min(s_idx.flatten()),
                                                      device=s_prompt_id.device)])
                    s_id_counts = torch.cat(
                        [s_id_counts, torch.full((self.s_pool_size - s_id_counts.shape[0],), 0, device=s_id_counts.device)])
                _, s_major_idx = torch.topk(s_id_counts, k=self.s_top_k)  # top_k
                s_major_prompt_id = s_prompt_id[s_major_idx]  # top_k
                # expand to batch
                s_idx = s_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

                t_prompt_id, t_id_counts = torch.unique(t_idx, return_counts=True, sorted=True)
                if t_prompt_id.shape[0] < self.t_pool_size:
                    t_prompt_id = torch.cat([t_prompt_id,
                                           torch.full((self.t_pool_size - t_prompt_id.shape[0],), torch.min(t_idx.flatten()),
                                                      device=t_prompt_id.device)])
                    t_id_counts = torch.cat(
                        [t_id_counts, torch.full((self.t_pool_size - t_id_counts.shape[0],), 0, device=t_id_counts.device)])
                _, t_major_idx = torch.topk(t_id_counts, k=self.t_top_k)  # top_k
                t_major_prompt_id = t_prompt_id[t_major_idx]  # top_k
                # expand to batch
                t_idx = t_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k


            if s_prompt_mask is not None:
                s_idx = s_prompt_mask  # B, top_k
            if t_prompt_mask is not None:
                t_idx = t_prompt_mask  # B, top_k


            out['s_prompt_idx'] = s_idx
            out['t_prompt_idx'] = t_idx
            if self.use_prefix_tune_for_e_prompt:
                s_batched_prompt_raw = self.s_prompt[:, s_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, s_top_k, length, num_heads, heads_embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,s_top_k * length, heads_embed_dim
                )
                t_batched_prompt_raw = self.t_prompt[:, t_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, t_top_k, length, num_heads, heads_embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,t_top_k * length, heads_embed_dim
                )
            else:
                s_batched_prompt_raw = self.s_prompt[:, s_idx]
                num_layers, batch_size, s_top_k, length, embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers, batch_size, s_top_k * length, embed_dim
                )
                t_batched_prompt_raw = self.t_prompt[:, t_idx]
                num_layers, batch_size, t_top_k, length, embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers, batch_size, t_top_k * length, embed_dim
                )

            s_batched_key_norm = s_prompt_key_norm[s_idx]  # B, top_k, C
            t_batched_key_norm = t_prompt_key_norm[t_idx]

            out['s_selected_key'] = s_batched_key_norm
            out['s_prompt_key_norm'] = s_prompt_key_norm
            out['t_selected_key'] = t_batched_key_norm
            out['t_prompt_key_norm'] = t_prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            s_sim = s_batched_key_norm * x_embed_norm  # B, top_k, C
            s_reduce_sim = torch.sum(s_sim) / x_embed_mean.shape[0]  # Scalar
            t_sim = t_batched_key_norm * x_embed_norm  # B, top_k, C
            t_reduce_sim = torch.sum(t_sim) / x_embed_mean.shape[0]  # Scalar

            out['s_reduce_sim'] = s_reduce_sim
            out['t_reduce_sim'] = t_reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['s_batched_prompt'] = s_batched_prompt
        out['t_batched_prompt'] = t_batched_prompt

        return out


class SPrompt_meta(nn.Module):
    def __init__(self,  embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, s_length=5,s_pool_size=None, s_top_k=None,t_length=5,t_pool_size=None,t_top_k=None,m_length=5, m_pool_size=None,m_top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, use_embed_for_s_prompt=False):
        super().__init__()

        self.s_length = s_length
        self.t_length = t_length
        self.m_length = m_length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.m_pool_size =m_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k
        self.m_top_k = m_top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_embed_for_s_prompt = use_embed_for_s_prompt


        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    s_prompt_pool_shape = (self.num_layers*2, self.s_pool_size, self.s_length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.s_prompt = nn.Parameter(torch.randn(
                            s_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.s_prompt, -1, 1)

                    t_prompt_pool_shape = (self.num_layers*2, self.t_pool_size, self.t_length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.t_prompt = nn.Parameter(torch.randn(
                            t_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.t_prompt, -1, 1)

                    m_prompt_pool_shape = (self.num_layers * 2, self.m_pool_size, self.m_length,
                                           self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.m_prompt = nn.Parameter(torch.zeros(m_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.m_prompt = nn.Parameter(torch.randn(
                            m_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.m_prompt, -1, 1)

            else:
                # if self.use_embed_for_s_prompt:
                #     prompt_pool_shape = (self.pool_size,self.length,embed_dim)
                #     self.prompt = torch.embedding()
                s_prompt_pool_shape = (self.num_layers, self.s_pool_size, self.s_length, embed_dim)
                t_prompt_pool_shape = (self.num_layers,self.t_pool_size,self.t_length,embed_dim)
                m_prompt_pool_shape = (self.num_layers,self.m_pool_size,self.m_length,embed_dim)
                if prompt_init == 'zero':
                    self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                    self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                    self.m_prompt = nn.Parameter(torch.zeros(m_prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.s_prompt = nn.Parameter(torch.randn(s_prompt_pool_shape))
                    self.t_prompt = nn.Parameter(torch.randn(t_prompt_pool_shape))
                    self.m_prompt = nn.Parameter(torch.randn(m_prompt_pool_shape))
                    nn.init.uniform_(self.s_prompt, -1, 1)
                    nn.init.uniform_(self.t_prompt, -1, 1)
                    nn.init.uniform_(self.m_prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            s_key_shape = (s_pool_size, embed_dim)
            t_key_shape = (t_pool_size, embed_dim)
            m_key_shape = (m_pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.s_prompt_key = nn.Parameter(torch.zeros(s_key_shape))
                self.t_prompt_key = nn.Parameter(torch.zeros(t_key_shape))
                self.m_prompt_key = nn.Parameter(torch.zeros(m_key_shape))
            elif prompt_key_init == 'uniform':
                self.s_prompt_key = nn.Parameter(torch.randn(s_key_shape))
                nn.init.uniform_(self.s_prompt_key, -1, 1)
                self.t_prompt_key = nn.Parameter(torch.randn(t_key_shape))
                nn.init.uniform_(self.t_prompt_key, -1, 1)
                self.m_prompt_key = nn.Parameter(torch.randn(m_key_shape))
                nn.init.uniform_(self.m_prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            s_prompt_mean = torch.mean(self.s_prompt, dim=[0, 2])
            self.s_prompt_key = s_prompt_mean
            t_prompt_mean = torch.mean(self.t_prompt, dim=[0, 2])
            self.t_prompt_key = t_prompt_mean
            m_prompt_mean = torch.mean(self.m_prompt, dim=[0, 2])
            self.m_prompt_key = m_prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, s_prompt_mask=None,t_prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            s_prompt_key_norm = self.l2_normalize(self.s_prompt_key, dim=-1)  # Pool_size, C
            t_prompt_key_norm = self.l2_normalize(self.t_prompt_key, dim=-1)  # Pool_size, C
            m_prompt_key_norm = self.l2_normalize(self.m_prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            s_similarity = torch.matmul(s_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            s_similarity = s_similarity.t()  # B, pool_size
            t_similarity = torch.matmul(t_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            t_similarity = t_similarity.t()  # B, pool_size
            m_similarity = torch.matmul(m_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            m_similarity = m_similarity.t()  # B, pool_size

            (s_similarity_top_k, s_idx) = torch.topk(s_similarity, k=self.s_top_k, dim=1)  # B, top_k
            out['s_similarity'] = s_similarity
            (t_similarity_top_k, t_idx) = torch.topk(t_similarity, k=self.t_top_k, dim=1)  # B, top_k
            out['t_similarity'] = t_similarity
            (m_similarity_top_k, m_idx) = torch.topk(m_similarity, k=self.m_top_k, dim=1)  # B, top_k
            out['t_similarity'] = m_similarity

            if self.batchwise_prompt:
                s_prompt_id, s_id_counts = torch.unique(s_idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if s_prompt_id.shape[0] < self.s_pool_size:
                    s_prompt_id = torch.cat([s_prompt_id,
                                           torch.full((self.s_pool_size - s_prompt_id.shape[0],), torch.min(s_idx.flatten()),
                                                      device=s_prompt_id.device)])
                    s_id_counts = torch.cat(
                        [s_id_counts, torch.full((self.s_pool_size - s_id_counts.shape[0],), 0, device=s_id_counts.device)])
                _, s_major_idx = torch.topk(s_id_counts, k=self.s_top_k)  # top_k
                s_major_prompt_id = s_prompt_id[s_major_idx]  # top_k
                # expand to batch
                s_idx = s_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

                t_prompt_id, t_id_counts = torch.unique(t_idx, return_counts=True, sorted=True)
                if t_prompt_id.shape[0] < self.t_pool_size:
                    t_prompt_id = torch.cat([t_prompt_id,
                                           torch.full((self.t_pool_size - t_prompt_id.shape[0],), torch.min(t_idx.flatten()),
                                                      device=t_prompt_id.device)])
                    t_id_counts = torch.cat(
                        [t_id_counts, torch.full((self.t_pool_size - t_id_counts.shape[0],), 0, device=t_id_counts.device)])
                _, t_major_idx = torch.topk(t_id_counts, k=self.t_top_k)  # top_k
                t_major_prompt_id = t_prompt_id[t_major_idx]  # top_k
                # expand to batch
                t_idx = t_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

                m_prompt_id, m_id_counts = torch.unique(m_idx, return_counts=True, sorted=True)
                if m_prompt_id.shape[0] < self.m_pool_size:
                    m_prompt_id = torch.cat([m_prompt_id,
                                             torch.full((self.m_pool_size - m_prompt_id.shape[0],),
                                                        torch.min(m_idx.flatten()),
                                                        device=m_prompt_id.device)])
                    m_id_counts = torch.cat(
                        [m_id_counts,
                         torch.full((self.m_pool_size - m_id_counts.shape[0],), 0, device=m_id_counts.device)])
                _, m_major_idx = torch.topk(m_id_counts, k=self.m_top_k)  # top_k
                m_major_prompt_id = m_prompt_id[m_major_idx]  # top_k
                # expand to batch
                m_idx = m_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

            if s_prompt_mask is not None:
                s_idx = s_prompt_mask  # B, top_k
            if t_prompt_mask is not None:
                t_idx = t_prompt_mask  # B, top_k


            out['s_prompt_idx'] = s_idx
            out['t_prompt_idx'] = t_idx
            out['m_prompt_idx'] = m_idx
            if self.use_prefix_tune_for_e_prompt:
                s_batched_prompt_raw = self.s_prompt[:, s_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, s_top_k, length, num_heads, heads_embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,s_top_k * length, heads_embed_dim
                )
                t_batched_prompt_raw = self.t_prompt[:, t_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, t_top_k, length, num_heads, heads_embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,t_top_k * length, heads_embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, m_top_k, length, num_heads, heads_embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers_dual, batch_size, num_heads, m_top_k * length, heads_embed_dim
                )
            else:
                s_batched_prompt_raw = self.s_prompt[:, s_idx]
                num_layers, batch_size, s_top_k, length, embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers, batch_size, s_top_k * length, embed_dim
                )
                t_batched_prompt_raw = self.t_prompt[:, t_idx]
                num_layers, batch_size, t_top_k, length, embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers, batch_size, t_top_k * length, embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]
                num_layers, batch_size, m_top_k, length, embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers, batch_size, m_top_k * length, embed_dim
                )

            s_batched_key_norm = s_prompt_key_norm[s_idx]  # B, top_k, C
            t_batched_key_norm = t_prompt_key_norm[t_idx]
            m_batched_key_norm = m_prompt_key_norm[m_idx]

            out['s_selected_key'] = s_batched_key_norm
            out['s_prompt_key_norm'] = s_prompt_key_norm
            out['t_selected_key'] = t_batched_key_norm
            out['t_prompt_key_norm'] = t_prompt_key_norm
            out['m_selected_key'] = m_batched_key_norm
            out['m_prompt_key_norm'] = m_prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            s_sim = s_batched_key_norm * x_embed_norm  # B, top_k, C
            s_reduce_sim = torch.sum(s_sim) / x_embed_mean.shape[0]  # Scalar
            t_sim = t_batched_key_norm * x_embed_norm  # B, top_k, C
            t_reduce_sim = torch.sum(t_sim) / x_embed_mean.shape[0]  # Scalar
            m_sim = m_batched_key_norm * x_embed_norm  # B, top_k, C
            m_reduce_sim = torch.sum(m_sim) / x_embed_mean.shape[0]  # Scalar


            out['s_reduce_sim'] = s_reduce_sim
            out['t_reduce_sim'] = t_reduce_sim
            out['m_reduce_sim'] = m_reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['s_batched_prompt'] = s_batched_prompt
        out['t_batched_prompt'] = t_batched_prompt
        out['m_batched_prompt'] = m_batched_prompt
        return out

class SPrompt_Wo_type(nn.Module):
    def __init__(self,  embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, s_length=5,s_pool_size=None, s_top_k=None,m_length=5, m_pool_size=None,m_top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=True, num_heads=-1, same_key_value=False, use_embed_for_s_prompt=False):
        super().__init__()

        self.s_length = s_length
        self.m_length = m_length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.s_pool_size = s_pool_size
        self.m_pool_size =m_pool_size
        self.s_top_k = s_top_k
        self.m_top_k = m_top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_embed_for_s_prompt = use_embed_for_s_prompt


        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    s_prompt_pool_shape = (self.num_layers*2, self.s_pool_size, self.s_length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.s_prompt = nn.Parameter(torch.randn(
                            s_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.s_prompt, -1, 1)


                    m_prompt_pool_shape = (self.num_layers * 2, self.m_pool_size, self.m_length,
                                           self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.m_prompt = nn.Parameter(torch.zeros(m_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.m_prompt = nn.Parameter(torch.randn(
                            m_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.m_prompt, -1, 1)

            else:
                # if self.use_embed_for_s_prompt:
                #     prompt_pool_shape = (self.pool_size,self.length,embed_dim)
                #     self.prompt = torch.embedding()
                s_prompt_pool_shape = (self.num_layers, self.s_pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.s_prompt = nn.Parameter(torch.randn(s_prompt_pool_shape))
                    nn.init.uniform_(self.s_prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            s_key_shape = (s_pool_size, embed_dim)
            m_key_shape = (m_pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.s_prompt_key = nn.Parameter(torch.zeros(s_key_shape))
                self.m_prompt_key = nn.Parameter(torch.zeros(m_key_shape))
            elif prompt_key_init == 'uniform':
                self.s_prompt_key = nn.Parameter(torch.randn(s_key_shape))
                nn.init.uniform_(self.s_prompt_key, -1, 1)
                self.m_prompt_key = nn.Parameter(torch.randn(m_key_shape))
                nn.init.uniform_(self.m_prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            s_prompt_mean = torch.mean(self.s_prompt, dim=[0, 2])
            self.s_prompt_key = s_prompt_mean
            m_prompt_mean = torch.mean(self.m_prompt, dim=[0, 2])
            self.m_prompt_key = m_prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, s_prompt_mask=None,t_prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            s_prompt_key_norm = self.l2_normalize(self.s_prompt_key, dim=-1)  # Pool_size, C
            m_prompt_key_norm = self.l2_normalize(self.m_prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            s_similarity = torch.matmul(s_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            s_similarity = s_similarity.t()  # B, pool_size
            m_similarity = torch.matmul(m_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            m_similarity = m_similarity.t()  # B, pool_size

            (s_similarity_top_k, s_idx) = torch.topk(s_similarity, k=self.s_top_k, dim=1)  # B, top_k
            out['s_similarity'] = s_similarity
            (m_similarity_top_k, m_idx) = torch.topk(m_similarity, k=self.m_top_k, dim=1)  # B, top_k
            out['t_similarity'] = m_similarity

            if self.batchwise_prompt:
                s_prompt_id, s_id_counts = torch.unique(s_idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if s_prompt_id.shape[0] < self.s_pool_size:
                    s_prompt_id = torch.cat([s_prompt_id,
                                           torch.full((self.s_pool_size - s_prompt_id.shape[0],), torch.min(s_idx.flatten()),
                                                      device=s_prompt_id.device)])
                    s_id_counts = torch.cat(
                        [s_id_counts, torch.full((self.s_pool_size - s_id_counts.shape[0],), 0, device=s_id_counts.device)])
                _, s_major_idx = torch.topk(s_id_counts, k=self.s_top_k)  # top_k
                s_major_prompt_id = s_prompt_id[s_major_idx]  # top_k
                # expand to batch
                s_idx = s_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

                m_prompt_id, m_id_counts = torch.unique(m_idx, return_counts=True, sorted=True)
                if m_prompt_id.shape[0] < self.m_pool_size:
                    m_prompt_id = torch.cat([m_prompt_id,
                                             torch.full((self.m_pool_size - m_prompt_id.shape[0],),
                                                        torch.min(m_idx.flatten()),
                                                        device=m_prompt_id.device)])
                    m_id_counts = torch.cat(
                        [m_id_counts,
                         torch.full((self.m_pool_size - m_id_counts.shape[0],), 0, device=m_id_counts.device)])
                _, m_major_idx = torch.topk(m_id_counts, k=self.m_top_k)  # top_k
                m_major_prompt_id = m_prompt_id[m_major_idx]  # top_k
                # expand to batch
                m_idx = m_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

            if s_prompt_mask is not None:
                s_idx = s_prompt_mask  # B, top_k
            if t_prompt_mask is not None:
                t_idx = t_prompt_mask  # B, top_k


            out['s_prompt_idx'] = s_idx
            out['m_prompt_idx'] = m_idx
            if self.use_prefix_tune_for_e_prompt:
                s_batched_prompt_raw = self.s_prompt[:, s_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, s_top_k, length, num_heads, heads_embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,s_top_k * length, heads_embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, m_top_k, length, num_heads, heads_embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers_dual, batch_size, num_heads, m_top_k * length, heads_embed_dim
                )
            else:
                s_batched_prompt_raw = self.s_prompt[:, s_idx]
                num_layers, batch_size, s_top_k, length, embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers, batch_size, s_top_k * length, embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]
                num_layers, batch_size, m_top_k, length, embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers, batch_size, m_top_k * length, embed_dim
                )

            s_batched_key_norm = s_prompt_key_norm[s_idx]  # B, top_k, C
            m_batched_key_norm = m_prompt_key_norm[m_idx]

            out['s_selected_key'] = s_batched_key_norm
            out['s_prompt_key_norm'] = s_prompt_key_norm
            out['m_selected_key'] = m_batched_key_norm
            out['m_prompt_key_norm'] = m_prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            s_sim = s_batched_key_norm * x_embed_norm  # B, top_k, C
            s_reduce_sim = torch.sum(s_sim) / x_embed_mean.shape[0]  # Scalar
            m_sim = m_batched_key_norm * x_embed_norm  # B, top_k, C
            m_reduce_sim = torch.sum(m_sim) / x_embed_mean.shape[0]  # Scalar


            out['s_reduce_sim'] = s_reduce_sim
            out['m_reduce_sim'] = m_reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['s_batched_prompt'] = s_batched_prompt
        out['m_batched_prompt'] = m_batched_prompt
        return out

class SPrompt_Wo_system(nn.Module):
    def __init__(self,  embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, t_length=5,t_pool_size=None,t_top_k=None,m_length=5, m_pool_size=None,m_top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, use_embed_for_s_prompt=False):
        super().__init__()


        self.t_length = t_length
        self.m_length = m_length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key

        self.t_pool_size = t_pool_size
        self.m_pool_size =m_pool_size

        self.t_top_k = t_top_k
        self.m_top_k = m_top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_embed_for_s_prompt = use_embed_for_s_prompt


        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:

                    t_prompt_pool_shape = (self.num_layers*2, self.t_pool_size, self.t_length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.t_prompt = nn.Parameter(torch.randn(
                            t_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.t_prompt, -1, 1)

                    m_prompt_pool_shape = (self.num_layers * 2, self.m_pool_size, self.m_length,
                                           self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.m_prompt = nn.Parameter(torch.zeros(m_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.m_prompt = nn.Parameter(torch.randn(
                            m_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.m_prompt, -1, 1)

            else:
                # if self.use_embed_for_s_prompt:
                #     prompt_pool_shape = (self.pool_size,self.length,embed_dim)
                #     self.prompt = torch.embedding()

                t_prompt_pool_shape = (self.num_layers,self.t_pool_size,self.t_length,embed_dim)
                m_prompt_pool_shape = (self.num_layers,self.m_pool_size,self.m_length,embed_dim)
                if prompt_init == 'zero':

                    self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                    self.m_prompt = nn.Parameter(torch.zeros(m_prompt_pool_shape))
                elif prompt_init == 'uniform':

                    self.t_prompt = nn.Parameter(torch.randn(t_prompt_pool_shape))
                    self.m_prompt = nn.Parameter(torch.randn(m_prompt_pool_shape))

                    nn.init.uniform_(self.t_prompt, -1, 1)
                    nn.init.uniform_(self.m_prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:

            t_key_shape = (t_pool_size, embed_dim)
            m_key_shape = (m_pool_size, embed_dim)
            if prompt_key_init == 'zero':

                self.t_prompt_key = nn.Parameter(torch.zeros(t_key_shape))
                self.m_prompt_key = nn.Parameter(torch.zeros(m_key_shape))
            elif prompt_key_init == 'uniform':

                self.t_prompt_key = nn.Parameter(torch.randn(t_key_shape))
                nn.init.uniform_(self.t_prompt_key, -1, 1)
                self.m_prompt_key = nn.Parameter(torch.randn(m_key_shape))
                nn.init.uniform_(self.m_prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix

            t_prompt_mean = torch.mean(self.t_prompt, dim=[0, 2])
            self.t_prompt_key = t_prompt_mean
            m_prompt_mean = torch.mean(self.m_prompt, dim=[0, 2])
            self.m_prompt_key = m_prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed,t_prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")


            t_prompt_key_norm = self.l2_normalize(self.t_prompt_key, dim=-1)  # Pool_size, C
            m_prompt_key_norm = self.l2_normalize(self.m_prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C


            t_similarity = torch.matmul(t_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            t_similarity = t_similarity.t()  # B, pool_size
            m_similarity = torch.matmul(m_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            m_similarity = m_similarity.t()  # B, pool_size


            (t_similarity_top_k, t_idx) = torch.topk(t_similarity, k=self.t_top_k, dim=1)  # B, top_k
            out['t_similarity'] = t_similarity
            (m_similarity_top_k, m_idx) = torch.topk(m_similarity, k=self.m_top_k, dim=1)  # B, top_k
            out['t_similarity'] = m_similarity

            if self.batchwise_prompt:


                t_prompt_id, t_id_counts = torch.unique(t_idx, return_counts=True, sorted=True)
                if t_prompt_id.shape[0] < self.t_pool_size:
                    t_prompt_id = torch.cat([t_prompt_id,
                                           torch.full((self.t_pool_size - t_prompt_id.shape[0],), torch.min(t_idx.flatten()),
                                                      device=t_prompt_id.device)])
                    t_id_counts = torch.cat(
                        [t_id_counts, torch.full((self.t_pool_size - t_id_counts.shape[0],), 0, device=t_id_counts.device)])
                _, t_major_idx = torch.topk(t_id_counts, k=self.t_top_k)  # top_k
                t_major_prompt_id = t_prompt_id[t_major_idx]  # top_k
                # expand to batch
                t_idx = t_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

                m_prompt_id, m_id_counts = torch.unique(m_idx, return_counts=True, sorted=True)
                if m_prompt_id.shape[0] < self.m_pool_size:
                    m_prompt_id = torch.cat([m_prompt_id,
                                             torch.full((self.m_pool_size - m_prompt_id.shape[0],),
                                                        torch.min(m_idx.flatten()),
                                                        device=m_prompt_id.device)])
                    m_id_counts = torch.cat(
                        [m_id_counts,
                         torch.full((self.m_pool_size - m_id_counts.shape[0],), 0, device=m_id_counts.device)])
                _, m_major_idx = torch.topk(m_id_counts, k=self.m_top_k)  # top_k
                m_major_prompt_id = m_prompt_id[m_major_idx]  # top_k
                # expand to batch
                m_idx = m_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k


            if t_prompt_mask is not None:
                t_idx = t_prompt_mask  # B, top_k



            out['t_prompt_idx'] = t_idx
            out['m_prompt_idx'] = m_idx
            if self.use_prefix_tune_for_e_prompt:

                t_batched_prompt_raw = self.t_prompt[:, t_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, t_top_k, length, num_heads, heads_embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,t_top_k * length, heads_embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, m_top_k, length, num_heads, heads_embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers_dual, batch_size, num_heads, m_top_k * length, heads_embed_dim
                )
            else:

                t_batched_prompt_raw = self.t_prompt[:, t_idx]
                num_layers, batch_size, t_top_k, length, embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers, batch_size, t_top_k * length, embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]
                num_layers, batch_size, m_top_k, length, embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers, batch_size, m_top_k * length, embed_dim
                )


            t_batched_key_norm = t_prompt_key_norm[t_idx]
            m_batched_key_norm = m_prompt_key_norm[m_idx]


            out['t_selected_key'] = t_batched_key_norm
            out['t_prompt_key_norm'] = t_prompt_key_norm
            out['m_selected_key'] = m_batched_key_norm
            out['m_prompt_key_norm'] = m_prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C

            t_sim = t_batched_key_norm * x_embed_norm  # B, top_k, C
            t_reduce_sim = torch.sum(t_sim) / x_embed_mean.shape[0]  # Scalar
            m_sim = m_batched_key_norm * x_embed_norm  # B, top_k, C
            m_reduce_sim = torch.sum(m_sim) / x_embed_mean.shape[0]  # Scalar



            out['t_reduce_sim'] = t_reduce_sim
            out['m_reduce_sim'] = m_reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)


        out['t_batched_prompt'] = t_batched_prompt
        out['m_batched_prompt'] = m_batched_prompt
        return out


class SPrompt_mean(nn.Module):
    def __init__(self,  embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, s_length=5,s_pool_size=None, s_top_k=None,t_length=5,t_pool_size=None,t_top_k=None,m_length=5, m_pool_size=None,m_top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, use_embed_for_s_prompt=False):
        super().__init__()

        self.s_length = s_length
        self.t_length = t_length
        self.m_length = m_length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.m_pool_size =m_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k
        self.m_top_k = m_top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_embed_for_s_prompt = use_embed_for_s_prompt


        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                         self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    s_prompt_pool_shape = (self.num_layers*2, self.s_pool_size, self.s_length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.s_prompt = nn.Parameter(torch.randn(
                            s_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.s_prompt, -1, 1)

                    t_prompt_pool_shape = (self.num_layers*2, self.t_pool_size, self.t_length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.t_prompt = nn.Parameter(torch.randn(
                            t_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.t_prompt, -1, 1)

                    m_prompt_pool_shape = (self.num_layers * 2, self.m_pool_size, self.m_length,
                                           self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.m_prompt = nn.Parameter(torch.zeros(m_prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.m_prompt = nn.Parameter(torch.randn(
                            m_prompt_pool_shape))  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.m_prompt, -1, 1)

            else:
                # if self.use_embed_for_s_prompt:
                #     prompt_pool_shape = (self.pool_size,self.length,embed_dim)
                #     self.prompt = torch.embedding()
                s_prompt_pool_shape = (self.num_layers, self.s_pool_size, self.s_length, embed_dim)
                t_prompt_pool_shape = (self.num_layers,self.t_pool_size,self.t_length,embed_dim)
                m_prompt_pool_shape = (self.num_layers,self.m_pool_size,self.m_length,embed_dim)
                if prompt_init == 'zero':
                    self.s_prompt = nn.Parameter(torch.zeros(s_prompt_pool_shape))
                    self.t_prompt = nn.Parameter(torch.zeros(t_prompt_pool_shape))
                    self.m_prompt = nn.Parameter(torch.zeros(m_prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.s_prompt = nn.Parameter(torch.randn(s_prompt_pool_shape))
                    self.t_prompt = nn.Parameter(torch.randn(t_prompt_pool_shape))
                    self.m_prompt = nn.Parameter(torch.randn(m_prompt_pool_shape))
                    nn.init.uniform_(self.s_prompt, -1, 1)
                    nn.init.uniform_(self.t_prompt, -1, 1)
                    nn.init.uniform_(self.m_prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            s_key_shape = (s_pool_size, embed_dim)
            t_key_shape = (t_pool_size, embed_dim)
            m_key_shape = (m_pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.s_prompt_key = nn.Parameter(torch.zeros(s_key_shape))
                self.t_prompt_key = nn.Parameter(torch.zeros(t_key_shape))
                self.m_prompt_key = nn.Parameter(torch.zeros(m_key_shape))
            elif prompt_key_init == 'uniform':
                self.s_prompt_key = nn.Parameter(torch.randn(s_key_shape))
                nn.init.uniform_(self.s_prompt_key, -1, 1)
                self.t_prompt_key = nn.Parameter(torch.randn(t_key_shape))
                nn.init.uniform_(self.t_prompt_key, -1, 1)
                self.m_prompt_key = nn.Parameter(torch.randn(m_key_shape))
                nn.init.uniform_(self.m_prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            s_prompt_mean = torch.mean(self.s_prompt, dim=[0, 2])
            self.s_prompt_key = s_prompt_mean
            t_prompt_mean = torch.mean(self.t_prompt, dim=[0, 2])
            self.t_prompt_key = t_prompt_mean
            m_prompt_mean = torch.mean(self.m_prompt, dim=[0, 2])
            self.m_prompt_key = m_prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, s_prompt_mask=None,t_prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            s_prompt_key_norm = self.l2_normalize(self.s_prompt_key, dim=-1)  # Pool_size, C
            t_prompt_key_norm = self.l2_normalize(self.t_prompt_key, dim=-1)  # Pool_size, C
            m_prompt_key_norm = self.l2_normalize(self.m_prompt_key, dim=-1)  # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

            s_similarity = torch.matmul(s_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            s_similarity = s_similarity.t()  # B, pool_size
            t_similarity = torch.matmul(t_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            t_similarity = t_similarity.t()  # B, pool_size
            m_similarity = torch.matmul(m_prompt_key_norm, x_embed_norm.t())  # pool_size, B or Pool_size, #class, B
            m_similarity = m_similarity.t()  # B, pool_size

            (s_similarity_top_k, s_idx) = torch.topk(s_similarity, k=self.s_top_k, dim=1)  # B, top_k
            out['s_similarity'] = s_similarity
            (t_similarity_top_k, t_idx) = torch.topk(t_similarity, k=self.t_top_k, dim=1)  # B, top_k
            out['t_similarity'] = t_similarity
            (m_similarity_top_k, m_idx) = torch.topk(m_similarity, k=self.m_top_k, dim=1)  # B, top_k
            out['t_similarity'] = m_similarity

            if self.batchwise_prompt:
                s_prompt_id, s_id_counts = torch.unique(s_idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if s_prompt_id.shape[0] < self.s_pool_size:
                    s_prompt_id = torch.cat([s_prompt_id,
                                           torch.full((self.s_pool_size - s_prompt_id.shape[0],), torch.min(s_idx.flatten()),
                                                      device=s_prompt_id.device)])
                    s_id_counts = torch.cat(
                        [s_id_counts, torch.full((self.s_pool_size - s_id_counts.shape[0],), 0, device=s_id_counts.device)])
                _, s_major_idx = torch.topk(s_id_counts, k=self.s_top_k)  # top_k
                s_major_prompt_id = s_prompt_id[s_major_idx]  # top_k
                # expand to batch
                s_idx = s_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

                t_prompt_id, t_id_counts = torch.unique(t_idx, return_counts=True, sorted=True)
                if t_prompt_id.shape[0] < self.t_pool_size:
                    t_prompt_id = torch.cat([t_prompt_id,
                                           torch.full((self.t_pool_size - t_prompt_id.shape[0],), torch.min(t_idx.flatten()),
                                                      device=t_prompt_id.device)])
                    t_id_counts = torch.cat(
                        [t_id_counts, torch.full((self.t_pool_size - t_id_counts.shape[0],), 0, device=t_id_counts.device)])
                _, t_major_idx = torch.topk(t_id_counts, k=self.t_top_k)  # top_k
                t_major_prompt_id = t_prompt_id[t_major_idx]  # top_k
                # expand to batch
                t_idx = t_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

                m_prompt_id, m_id_counts = torch.unique(m_idx, return_counts=True, sorted=True)
                if m_prompt_id.shape[0] < self.m_pool_size:
                    m_prompt_id = torch.cat([m_prompt_id,
                                             torch.full((self.m_pool_size - m_prompt_id.shape[0],),
                                                        torch.min(m_idx.flatten()),
                                                        device=m_prompt_id.device)])
                    m_id_counts = torch.cat(
                        [m_id_counts,
                         torch.full((self.m_pool_size - m_id_counts.shape[0],), 0, device=m_id_counts.device)])
                _, m_major_idx = torch.topk(m_id_counts, k=self.m_top_k)  # top_k
                m_major_prompt_id = m_prompt_id[m_major_idx]  # top_k
                # expand to batch
                m_idx = m_major_prompt_id.expand(x_embed_mean.shape[0], -1).contiguous()  # B, top_k

            if s_prompt_mask is not None:
                s_idx = s_prompt_mask  # B, top_k
            if t_prompt_mask is not None:
                t_idx = t_prompt_mask  # B, top_k


            out['s_prompt_idx'] = s_idx
            out['t_prompt_idx'] = t_idx
            out['m_prompt_idx'] = m_idx
            if self.use_prefix_tune_for_e_prompt:
                s_batched_prompt_raw = self.s_prompt.unsqueeze(1).expand(-1, x_embed_mean.shape[0], -1, -1, -1,-1)
                s_batched_prompt_raw = torch.mean(s_batched_prompt_raw,dim=2).unsqueeze(2)  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, s_top_k, length, num_heads, heads_embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,s_top_k * length, heads_embed_dim
                )

                t_batched_prompt_raw = self.t_prompt.unsqueeze(1).expand(-1, x_embed_mean.shape[0], -1, -1, -1, -1)
                t_batched_prompt_raw = torch.mean(t_batched_prompt_raw, dim=2).unsqueeze(
                    2)  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, t_top_k, length, num_heads, heads_embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers_dual,batch_size, num_heads,t_top_k * length, heads_embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]  # num_layers, B, top_k, length, C
                num_layers_dual, batch_size, m_top_k, length, num_heads, heads_embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers_dual, batch_size, num_heads, m_top_k * length, heads_embed_dim
                )
            else:
                s_batched_prompt_raw = self.s_prompt[:, s_idx]
                num_layers, batch_size, s_top_k, length, embed_dim = s_batched_prompt_raw.shape
                s_batched_prompt = s_batched_prompt_raw.reshape(
                    num_layers, batch_size, s_top_k * length, embed_dim
                )
                t_batched_prompt_raw = self.t_prompt[:, t_idx]
                num_layers, batch_size, t_top_k, length, embed_dim = t_batched_prompt_raw.shape
                t_batched_prompt = t_batched_prompt_raw.reshape(
                    num_layers, batch_size, t_top_k * length, embed_dim
                )
                m_batched_prompt_raw = self.m_prompt[:, m_idx]
                num_layers, batch_size, m_top_k, length, embed_dim = m_batched_prompt_raw.shape
                m_batched_prompt = m_batched_prompt_raw.reshape(
                    num_layers, batch_size, m_top_k * length, embed_dim
                )

            s_batched_key_norm = s_prompt_key_norm[s_idx]  # B, top_k, C
            t_batched_key_norm = t_prompt_key_norm[t_idx]
            m_batched_key_norm = m_prompt_key_norm[m_idx]

            out['s_selected_key'] = s_batched_key_norm
            out['s_prompt_key_norm'] = s_prompt_key_norm
            out['t_selected_key'] = t_batched_key_norm
            out['t_prompt_key_norm'] = t_prompt_key_norm
            out['m_selected_key'] = m_batched_key_norm
            out['m_prompt_key_norm'] = m_prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            s_sim = s_batched_key_norm * x_embed_norm  # B, top_k, C
            s_reduce_sim = torch.sum(s_sim) / x_embed_mean.shape[0]  # Scalar
            t_sim = t_batched_key_norm * x_embed_norm  # B, top_k, C
            t_reduce_sim = torch.sum(t_sim) / x_embed_mean.shape[0]  # Scalar
            m_sim = m_batched_key_norm * x_embed_norm  # B, top_k, C
            m_reduce_sim = torch.sum(m_sim) / x_embed_mean.shape[0]  # Scalar


            out['s_reduce_sim'] = s_reduce_sim
            out['t_reduce_sim'] = t_reduce_sim
            out['m_reduce_sim'] = m_reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                         self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(
                            torch.randn(prompt_pool_shape))  # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['s_batched_prompt'] = s_batched_prompt
        out['t_batched_prompt'] = t_batched_prompt
        out['m_batched_prompt'] = m_batched_prompt
        return out
