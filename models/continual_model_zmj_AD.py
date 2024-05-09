import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput,SequenceClassifierOutput


from prompt_zmj import SPrompt,SPrompt_meta,SPrompt_mul,SPrompt_Wo_type,SPrompt_Wo_system,SPrompt_mean





class RobertaPromptForLogParsing(nn.Module):
    def __init__(self, num_labels,model_path,hidden_dropout_prob=0.1,
                 use_prompt_mask=False,embedding_key='cls', prompt_init='uniform',
            batchwise_prompt=False, prompt_key_init='uniform', head_type='token',
                 use_embed_for_G_prompt=False,
            g_prompt_length=None,  s_prompt_length=None, s_prompt_pool=False, s_prompt_key=False, s_pool_size=None,
            s_top_k=None,s_prompt_layer_idx=None, use_prefix_tune_for_s_prompt=False, same_key_value=False,):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.embeddings = self.roberta.embeddings
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        self.use_embed_for_G_prompt = use_embed_for_G_prompt
        self.g_prompt_length = g_prompt_length
        if self.use_embed_for_G_prompt:
            self.g_prompt_tokens = torch.arange(self.g_prompt_length).long()
            self.g_prompt_encoder = torch.nn.Embedding(self.g_prompt_length, self.roberta.config.hidden_size)
        else:
            g_prompt_shape = (1, g_prompt_length, self.roberta.config.hidden_size)
            self.one_g_prompts = nn.Parameter(torch.randn(g_prompt_shape))

        self.s_prompt_length = s_prompt_length
        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.s_top_k = s_top_k

        self.s_prompt = SPrompt(length=s_prompt_length, embed_dim=self.roberta.config.hidden_size, embedding_key=embedding_key, prompt_init=prompt_init,
                    prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, pool_size=s_pool_size, top_k=s_top_k, batchwise_prompt=batchwise_prompt,
                    prompt_key_init=prompt_key_init, num_layers=len(s_prompt_layer_idx), use_prefix_tune_for_e_prompt=False,
                    num_heads=self.n_head, same_key_value=same_key_value)


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def get_G_prompt(self,batch_size):
        if self.use_embed_for_G_prompt:
            g_prompt_tokens = self.g_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
            g_prompts = self.prefix_encoder(g_prompt_tokens)
        else:
            g_prompts = self.one_g_prompts.expand(batch_size, self.g_prompt_length,768).to(self.roberta.device)
        return g_prompts

    def get_T_prompt(self,batch_size):
        pass

    def get_S_prompt(self,batch_size):
        s_prompt_tokens = self.s_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        s_prompts = self.prefix_encoder(s_prompt_tokens)
        return s_prompts

    def get_M_prompt(self,batch_size):
        pass

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )



        g_prompts = self.get_G_prompt(batch_size=batch_size)
        if train:
            start = (task_id+1) * self.s_prompt.top_k
            end = (task_id+1 + 1) * self.s_prompt.top_k
            single_prompt_mask = torch.arange(start, end).to(raw_embedding.device)
            prompt_mask = single_prompt_mask.unsqueeze(0).expand(raw_embedding.shape[0], -1)
            if end > self.s_prompt.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None
        res = self.s_prompt(raw_embedding, prompt_mask=prompt_mask, cls_features=cls_features)
        s_prompts = res['batched_prompt'].squeeze()


        inputs_embeds = torch.cat((g_prompts,s_prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.g_prompt_length+self.s_prompt_length).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask = attention_mask[:, self.g_prompt_length+self.s_prompt_length:].contiguous()

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # # active_logits = logits.view(-1, self.num_labels)
                # active_labels = torch.where(
                #     active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                # )
                # active_logits = logits[:,self.g_prompt_length+self.s_prompt_length:].contiguous().view(-1, self.num_labels)
                active_labels = labels.view(-1)
                active_logits = logits[:,self.g_prompt_length+self.s_prompt_length:].contiguous().view(-1, self.num_labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss - 1 * res['reduce_sim']
        else:
            active_logits = logits[:, self.g_prompt_length + self.s_prompt_length:]


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=active_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPromptForLogParsing_v2(nn.Module):
    def __init__(self, num_labels,model_path,hidden_dropout_prob=0.1,
                 use_prompt_mask=False,embedding_key='cls', prompt_init='uniform',
            batchwise_prompt=False, prompt_key_init='uniform', head_type='token',
                 use_embed_for_G_prompt=False,g_prompt_length=None,
                t_prompt_length=None,t_prompt_pool=False,t_prompt_key=False,t_pool_size=None,
                 s_prompt_length=None, s_prompt_pool=False, s_prompt_key=False, s_pool_size=None,
            s_top_k=None,s_prompt_layer_idx=None, use_prefix_tune_for_s_prompt=False, same_key_value=False,):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.embeddings = self.roberta.embeddings
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        self.use_embed_for_G_prompt = use_embed_for_G_prompt
        self.g_prompt_length = g_prompt_length
        if self.use_embed_for_G_prompt:
            self.g_prompt_tokens = torch.arange(self.g_prompt_length).long()
            self.g_prompt_encoder = torch.nn.Embedding(self.g_prompt_length, self.roberta.config.hidden_size)
        else:
            g_prompt_shape = (1, g_prompt_length, self.roberta.config.hidden_size)
            self.one_g_prompts = nn.Parameter(torch.randn(g_prompt_shape))

        self.s_prompt_length = s_prompt_length
        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.s_top_k = s_top_k

        self.s_prompt = SPrompt(length=s_prompt_length, embed_dim=self.roberta.config.hidden_size, embedding_key=embedding_key, prompt_init=prompt_init,
                    prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, pool_size=s_pool_size, top_k=s_top_k, batchwise_prompt=batchwise_prompt,
                    prompt_key_init=prompt_key_init, num_layers=len(s_prompt_layer_idx), use_prefix_tune_for_e_prompt=False,
                    num_heads=self.n_head, same_key_value=same_key_value)


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def get_G_prompt(self,batch_size):
        if self.use_embed_for_G_prompt:
            g_prompt_tokens = self.g_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
            g_prompts = self.prefix_encoder(g_prompt_tokens)
        else:
            g_prompts = self.one_g_prompts.expand(batch_size, 5,768).to(self.roberta.device)
        return g_prompts

    def get_T_prompt(self,batch_size):
        pass

    def get_S_prompt(self,batch_size):
        s_prompt_tokens = self.s_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        s_prompts = self.prefix_encoder(s_prompt_tokens)
        return s_prompts

    def get_M_prompt(self,batch_size):
        pass



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )



        g_prompts = self.get_G_prompt(batch_size=batch_size)

        start = (task_id+1) * self.s_prompt.top_k
        end = (task_id+1 + 1) * self.s_prompt.top_k
        single_prompt_mask = torch.arange(start, end).to(raw_embedding.device)
        prompt_mask = single_prompt_mask.unsqueeze(0).expand(raw_embedding.shape[0], -1)
        res = self.s_prompt(raw_embedding, prompt_mask=prompt_mask, cls_features=cls_features)
        s_prompts = res['batched_prompt'].squeeze()


        inputs_embeds = torch.cat((g_prompts,s_prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.g_prompt_length+self.s_prompt_length).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask = attention_mask[:, self.g_prompt_length+self.s_prompt_length:].contiguous()

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                active_logits = logits[:,self.g_prompt_length+self.s_prompt_length:].contiguous().view(-1, self.num_labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss - 1 * res['reduce_sim']
        else:
            active_logits = logits[:, self.g_prompt_length + self.s_prompt_length:]


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=active_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class PrefixEncoder_clean(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, pre_seq_len,hidden_size,prefix_hidden_size,num_hidden_layers,prefix_projection=False):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size,prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

class RobertaPrefixForLogParsing_onlyprompt(nn.Module):
    def __init__(self, model_path,pre_seq_len=10,num_labels=2,hidden_dropout_prob=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)
        # self.init_weights()

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = pre_seq_len
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
                                                         self.n_layer)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask = attention_mask[:, self.pre_seq_len:].contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLogParsing(nn.Module):
    def __init__(self, model_path,num_labels=2,hidden_dropout_prob=0.1,
        use_prompt_mask = False, embedding_key = 'cls', prompt_init = 'uniform',
        batchwise_prompt = False, prompt_key_init = 'uniform', head_type = 'token',
        g_prompt_length = None, s_prompt_length = None, s_prompt_pool = False, s_prompt_key = False, s_pool_size = None,
        s_top_k = None, s_prompt_layer_idx = None, use_prefix_tune_for_s_prompt = True, same_key_value = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.g_prefix_length = g_prompt_length
        self.s_prefix_length = s_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
        #                                                  self.n_layer)


        g_prompt_shape = (self.g_prefix_length, self.n_layer * 2, self.n_head, self.n_embd)
        self.g_prefix = nn.Parameter(torch.randn(g_prompt_shape))


        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.s_top_k = s_top_k

        self.s_prompt = SPrompt(length=s_prompt_length, embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, pool_size=s_pool_size,
                                top_k=s_top_k, batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                use_prefix_tune_for_e_prompt=True,
                                num_heads=self.n_head, same_key_value=same_key_value)


        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size,s_past_key_values):
        g_past_key_values = self.g_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(self.roberta.device)
        # g_past_key_values = self.dropout(g_past_key_values)
        g_past_key_values = g_past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.cat((g_past_key_values,s_past_key_values),dim=3)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            start = (task_id) * self.s_prompt.top_k
            end = (task_id + 1) * self.s_prompt.top_k
            single_prompt_mask = torch.arange(start, end).to(input_ids.device)
            prompt_mask = single_prompt_mask.unsqueeze(0).expand(input_ids.shape[0], -1)
            if end > self.s_prompt.pool_size:
                prompt_mask = None
        else:
            prompt_mask = None
        res = self.s_prompt(None, prompt_mask=prompt_mask, cls_features=cls_features)
        s_past_key_values = res['batched_prompt'].squeeze()
        past_key_values = self.get_G_prompt(batch_size=batch_size,s_past_key_values=s_past_key_values)



        # past_key_values = torch.cat((g_past_key_values,s_past_key_values),dim=3)


        prefix_attention_mask = torch.ones(batch_size, self.g_prefix_length+self.s_prefix_length).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask = attention_mask[:, self.g_prefix_length+self.s_prefix_length:].contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLogAD_type_prompt(nn.Module):
    def __init__(self, model_path, num_labels=2, hidden_dropout_prob=0.1,
                 use_prompt_mask=False, embedding_key='cls', prompt_init='uniform',
                 batchwise_prompt=False, prompt_key_init='uniform', head_type='token',
                 g_prompt_length=None, s_prompt_length=None, t_prompt_length=None,
                 s_prompt_pool=False, s_prompt_key=False, s_pool_size=None,
                 s_top_k=None, t_pool_size=None, t_top_k=None,
                 s_prompt_layer_idx=None, use_prefix_tune_for_s_prompt=True, same_key_value=False,
                 ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.g_prefix_length = g_prompt_length
        self.s_prefix_length = s_prompt_length
        self.t_prefix_length = t_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
        #                                                  self.n_layer)

        g_prompt_shape = (self.g_prefix_length, self.n_layer * 2, self.n_head, self.n_embd)
        self.g_prefix = nn.Parameter(torch.randn(g_prompt_shape))

        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k

        self.s_prompt = SPrompt_mul(length=s_prompt_length, embed_dim=self.roberta.config.hidden_size,
                                    embedding_key=embedding_key, prompt_init=prompt_init,
                                    prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, s_pool_size=s_pool_size,
                                    s_top_k=s_top_k, t_pool_size=t_pool_size, t_top_k=t_top_k,
                                    batchwise_prompt=batchwise_prompt,
                                    prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                    use_prefix_tune_for_e_prompt=True,
                                    num_heads=self.n_head, same_key_value=same_key_value)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size, s_past_key_values, t_past_key_values):
        g_past_key_values = self.g_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(self.roberta.device)
        # g_past_key_values = self.dropout(g_past_key_values)
        g_past_key_values = g_past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.cat((g_past_key_values, t_past_key_values, s_past_key_values), dim=3)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            type_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            start = task_id * self.s_prompt.s_top_k
            end = (task_id + 1) * self.s_prompt.s_top_k
            single_prompt_mask = torch.arange(start, end).to(input_ids.device)
            s_prompt_mask = single_prompt_mask.unsqueeze(0).expand(input_ids.shape[0], -1)
            if end > self.s_prompt.s_pool_size:
                s_prompt_mask = None
            t_prompt_mask = torch.arange(type_id, type_id + 1).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            s_prompt_mask = None
            t_prompt_mask = None
        res = self.s_prompt(None, s_prompt_mask=s_prompt_mask, t_prompt_mask=t_prompt_mask, cls_features=cls_features)
        s_past_key_values = res['s_batched_prompt'].squeeze()
        t_past_key_values = res['t_batched_prompt'].squeeze()
        past_key_values = self.get_G_prompt(batch_size=batch_size, s_past_key_values=s_past_key_values,
                                            t_past_key_values=t_past_key_values)


        prefix_attention_mask = torch.ones(batch_size,
                                           self.g_prefix_length + self.t_prefix_length + self.s_prefix_length).to(
            self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )


        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLogAD_type_meta_prompt(nn.Module):
    def __init__(self, model_path,num_labels=2,hidden_dropout_prob=0.1,
        use_prompt_mask = False, embedding_key = 'cls', prompt_init = 'uniform',
        batchwise_prompt = False, prompt_key_init = 'uniform', head_type = 'token',
        g_prompt_length = None, s_prompt_length = None, t_prompt_length=None,
        s_prompt_pool = False, s_prompt_key = False, s_pool_size = None,
        s_top_k = None, t_pool_size=None,t_top_k=None, m_prompt_length=None, m_pool_size=None, m_top_k=None,
        s_prompt_layer_idx = None, use_prefix_tune_for_s_prompt = True, same_key_value = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.g_prefix_length = g_prompt_length
        self.s_prefix_length = s_prompt_length
        self.t_prefix_length = t_prompt_length
        self.m_prefix_length = m_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
        #                                                  self.n_layer)


        g_prompt_shape = (self.g_prefix_length, self.n_layer * 2, self.n_head, self.n_embd)
        self.g_prefix = nn.Parameter(torch.randn(g_prompt_shape))


        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.m_pool_size = m_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k
        self.m_top_k = m_top_k

        self.s_prompt = SPrompt_meta(s_length=s_prompt_length, t_length=t_prompt_length,m_length=m_prompt_length,embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, s_pool_size=s_pool_size,
                                s_top_k=s_top_k, t_pool_size=t_pool_size,t_top_k=t_top_k,
                                m_pool_size=m_pool_size,m_top_k=m_top_k,
                                batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                use_prefix_tune_for_e_prompt=use_prefix_tune_for_s_prompt,
                                num_heads=self.n_head, same_key_value=same_key_value)


        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size,s_past_key_values,t_past_key_values,m_past_key_values):
        g_past_key_values = self.g_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(self.roberta.device)
        # g_past_key_values = self.dropout(g_past_key_values)
        g_past_key_values = g_past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.cat((g_past_key_values,t_past_key_values,s_past_key_values,m_past_key_values),dim=3)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            type_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            start = task_id * self.s_prompt.s_top_k
            end = (task_id + 1) * self.s_prompt.s_top_k
            single_prompt_mask = torch.arange(start, end).to(input_ids.device)
            s_prompt_mask = single_prompt_mask.unsqueeze(0).expand(input_ids.shape[0], -1)
            if end > self.s_prompt.s_pool_size:
                s_prompt_mask = None
            t_prompt_mask = torch.arange(type_id, type_id + 1).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            s_prompt_mask = None
            t_prompt_mask = None
        res = self.s_prompt(None, s_prompt_mask=s_prompt_mask, t_prompt_mask=t_prompt_mask,cls_features=cls_features)
        s_past_key_values = res['s_batched_prompt'].squeeze()
        t_past_key_values = res['t_batched_prompt'].squeeze()
        m_past_key_values = res['m_batched_prompt'].squeeze()
        if train==False:
            s_prompt_idx = res['s_prompt_idx']
            t_prompt_idx = res['t_prompt_idx']

        else:
            s_prompt_idx = None
            t_prompt_idx =None

        past_key_values = self.get_G_prompt(batch_size=batch_size,s_past_key_values=s_past_key_values,
                                            t_past_key_values=t_past_key_values,m_past_key_values=m_past_key_values)


        prefix_attention_mask = torch.ones(batch_size, self.g_prefix_length+self.t_prefix_length*self.t_top_k+self.s_prefix_length*self.s_top_k+self.m_prefix_length*self.m_top_k).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if train:
            loss = loss - 1 * res['t_reduce_sim'] - 1 * res['s_reduce_sim']
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        if train:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ),t_prompt_idx,s_prompt_idx

class RobertaPrefixForLogAD_meta_prompt(nn.Module):
    def __init__(self, model_path,num_labels=2,hidden_dropout_prob=0.1,
        use_prompt_mask = False, embedding_key = 'cls', prompt_init = 'uniform',
        batchwise_prompt = False, prompt_key_init = 'uniform', head_type = 'token',
        g_prompt_length = None, s_prompt_length = None,
        s_prompt_pool = False, s_prompt_key = False, s_pool_size = None,
        s_top_k = None, m_prompt_length=None, m_pool_size=None, m_top_k=None,
        s_prompt_layer_idx = None, use_prefix_tune_for_s_prompt = True, same_key_value = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.g_prefix_length = g_prompt_length
        self.s_prefix_length = s_prompt_length
        self.m_prefix_length = m_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
        #                                                  self.n_layer)


        g_prompt_shape = (self.g_prefix_length, self.n_layer * 2, self.n_head, self.n_embd)
        self.g_prefix = nn.Parameter(torch.randn(g_prompt_shape))


        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.m_pool_size = m_pool_size
        self.s_top_k = s_top_k
        self.m_top_k = m_top_k

        self.s_prompt = SPrompt_Wo_type(s_length=s_prompt_length, m_length=m_prompt_length,embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, s_pool_size=s_pool_size,
                                s_top_k=s_top_k,
                                m_pool_size=m_pool_size,m_top_k=m_top_k,
                                batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                use_prefix_tune_for_e_prompt=True,
                                num_heads=self.n_head, same_key_value=same_key_value)


        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size,s_past_key_values,m_past_key_values):
        g_past_key_values = self.g_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(self.roberta.device)
        # g_past_key_values = self.dropout(g_past_key_values)
        g_past_key_values = g_past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.cat((g_past_key_values,s_past_key_values,m_past_key_values),dim=3)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            type_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            start = task_id * self.s_prompt.s_top_k
            end = (task_id + 1) * self.s_prompt.s_top_k
            single_prompt_mask = torch.arange(start, end).to(input_ids.device)
            s_prompt_mask = single_prompt_mask.unsqueeze(0).expand(input_ids.shape[0], -1)
            if end > self.s_prompt.s_pool_size:
                s_prompt_mask = None
            t_prompt_mask = torch.arange(type_id, type_id + 1).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            s_prompt_mask = None
            t_prompt_mask = None
        res = self.s_prompt(None, s_prompt_mask=s_prompt_mask, t_prompt_mask=t_prompt_mask,cls_features=cls_features)
        s_past_key_values = res['s_batched_prompt'].squeeze()
        m_past_key_values = res['m_batched_prompt'].squeeze()
        past_key_values = self.get_G_prompt(batch_size=batch_size,s_past_key_values=s_past_key_values,
                                            m_past_key_values=m_past_key_values)


        prefix_attention_mask = torch.ones(batch_size, self.g_prefix_length+self.s_prefix_length*self.s_top_k+self.m_prefix_length*self.m_top_k).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLogAD_Wo_System(nn.Module):
    def __init__(self, model_path,num_labels=2,hidden_dropout_prob=0.1,
        use_prompt_mask = False, embedding_key = 'cls', prompt_init = 'uniform',
        batchwise_prompt = False, prompt_key_init = 'uniform', head_type = 'token',
        g_prompt_length = None,  t_prompt_length=None,
        s_prompt_pool = False, s_prompt_key = False, t_pool_size=None,t_top_k=None, m_prompt_length=None, m_pool_size=None, m_top_k=None,
        s_prompt_layer_idx = None, use_prefix_tune_for_s_prompt = True, same_key_value = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.g_prefix_length = g_prompt_length
        self.t_prefix_length = t_prompt_length
        self.m_prefix_length = m_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
        #                                                  self.n_layer)


        g_prompt_shape = (self.g_prefix_length, self.n_layer * 2, self.n_head, self.n_embd)
        self.g_prefix = nn.Parameter(torch.randn(g_prompt_shape))


        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key

        self.t_pool_size = t_pool_size
        self.m_pool_size = m_pool_size

        self.t_top_k = t_top_k
        self.m_top_k = m_top_k

        self.s_prompt =SPrompt_Wo_system( t_length=t_prompt_length,m_length=m_prompt_length,embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key,
                                 t_pool_size=t_pool_size,t_top_k=t_top_k,
                                m_pool_size=m_pool_size,m_top_k=m_top_k,
                                batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                use_prefix_tune_for_e_prompt=True,
                                num_heads=self.n_head, same_key_value=same_key_value)


        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size,t_past_key_values,m_past_key_values):
        g_past_key_values = self.g_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(self.roberta.device)
        # g_past_key_values = self.dropout(g_past_key_values)
        g_past_key_values = g_past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.cat((g_past_key_values,t_past_key_values,m_past_key_values),dim=3)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            type_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            t_prompt_mask = torch.arange(type_id, type_id + 1).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:

            t_prompt_mask = None
        res = self.s_prompt(None, t_prompt_mask=t_prompt_mask,cls_features=cls_features)
        t_past_key_values = res['t_batched_prompt'].squeeze()
        m_past_key_values = res['m_batched_prompt'].squeeze()
        past_key_values = self.get_G_prompt(batch_size=batch_size,
                                            t_past_key_values=t_past_key_values,m_past_key_values=m_past_key_values)


        prefix_attention_mask = torch.ones(batch_size, self.g_prefix_length+self.t_prefix_length*self.t_top_k+self.m_prefix_length*self.m_top_k).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLogAD_Wo_General(nn.Module):
    def __init__(self, model_path,num_labels=2,hidden_dropout_prob=0.1,
        use_prompt_mask = False, embedding_key = 'cls', prompt_init = 'uniform',
        batchwise_prompt = False, prompt_key_init = 'uniform', head_type = 'token',
        s_prompt_length = None, t_prompt_length=None,
        s_prompt_pool = False, s_prompt_key = False, s_pool_size = None,
        s_top_k = None, t_pool_size=None,t_top_k=None, m_prompt_length=None, m_pool_size=None, m_top_k=None,
        s_prompt_layer_idx = None, use_prefix_tune_for_s_prompt = True, same_key_value = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.s_prefix_length = s_prompt_length
        self.t_prefix_length = t_prompt_length
        self.m_prefix_length = m_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads




        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.m_pool_size = m_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k
        self.m_top_k = m_top_k

        self.s_prompt = SPrompt_meta(s_length=s_prompt_length, t_length=t_prompt_length,m_length=m_prompt_length,embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, s_pool_size=s_pool_size,
                                s_top_k=s_top_k, t_pool_size=t_pool_size,t_top_k=t_top_k,
                                m_pool_size=m_pool_size,m_top_k=m_top_k,
                                batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                use_prefix_tune_for_e_prompt=True,
                                num_heads=self.n_head, same_key_value=same_key_value)


        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size,s_past_key_values,t_past_key_values,m_past_key_values):

        past_key_values = torch.cat((t_past_key_values,s_past_key_values,m_past_key_values),dim=3)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            type_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            start = task_id * self.s_prompt.s_top_k
            end = (task_id + 1) * self.s_prompt.s_top_k
            single_prompt_mask = torch.arange(start, end).to(input_ids.device)
            s_prompt_mask = single_prompt_mask.unsqueeze(0).expand(input_ids.shape[0], -1)
            if end > self.s_prompt.s_pool_size:
                s_prompt_mask = None
            t_prompt_mask = torch.arange(type_id, type_id + 1).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            s_prompt_mask = None
            t_prompt_mask = None
        res = self.s_prompt(None, s_prompt_mask=s_prompt_mask, t_prompt_mask=t_prompt_mask,cls_features=cls_features)
        s_past_key_values = res['s_batched_prompt'].squeeze()
        t_past_key_values = res['t_batched_prompt'].squeeze()
        m_past_key_values = res['m_batched_prompt'].squeeze()
        past_key_values = self.get_G_prompt(batch_size=batch_size,s_past_key_values=s_past_key_values,
                                            t_past_key_values=t_past_key_values,m_past_key_values=m_past_key_values)


        prefix_attention_mask = torch.ones(batch_size, self.t_prefix_length*self.t_top_k+self.s_prefix_length*self.s_top_k+self.m_prefix_length*self.m_top_k).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLAD_onlyprompt(nn.Module):
    def __init__(self, model_path,pre_seq_len=10,num_labels=2,hidden_dropout_prob=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)
        # self.init_weights()

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = pre_seq_len
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
                                                         self.n_layer)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPromptForLogAD_type_meta_prompt(nn.Module):
    def __init__(self, num_labels,model_path,hidden_dropout_prob=0.1,
                 use_prompt_mask=False,embedding_key='cls', prompt_init='uniform',
            batchwise_prompt=False, prompt_key_init='uniform', head_type='token',
            use_G_prompt=True,use_embed_for_G_prompt=False,
            g_prompt_length=None,  s_prompt_length=None, t_prompt_length=None, m_prompt_length=None,
            s_prompt_pool=False, s_prompt_key=False, s_pool_size=None,
            s_top_k=None,t_pool_size=None,t_top_k=None,m_pool_size=None,m_top_k=None,
            s_prompt_layer_idx=None, use_prefix_tune_for_s_prompt=False, same_key_value=False,):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.embeddings = self.roberta.embeddings
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        self.use_G_prompt = use_G_prompt
        if use_G_prompt:
            self.use_embed_for_G_prompt = use_embed_for_G_prompt
            self.g_prompt_length = g_prompt_length
            if self.use_embed_for_G_prompt:
                self.g_prompt_tokens = torch.arange(self.g_prompt_length).long()
                self.g_prompt_encoder = torch.nn.Embedding(self.g_prompt_length, self.roberta.config.hidden_size)
            else:
                g_prompt_shape = (1, g_prompt_length, self.roberta.config.hidden_size)
                self.one_g_prompts = nn.Parameter(torch.randn(g_prompt_shape))
        else:
            self.g_prompt_length=0


        self.s_prompt_length = s_prompt_length
        self.t_prompt_length = t_prompt_length
        self.m_prompt_length = m_prompt_length
        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.m_pool_size = m_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k
        self.m_top_k = m_top_k
        self.s_prompt = SPrompt_meta(s_length=s_prompt_length, t_length=t_prompt_length,m_length=m_prompt_length,embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, s_pool_size=s_pool_size,
                                s_top_k=s_top_k, t_pool_size=t_pool_size,t_top_k=t_top_k,
                                m_pool_size=m_pool_size,m_top_k=m_top_k,
                                batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=len(s_prompt_layer_idx),
                                use_prefix_tune_for_e_prompt=False,
                                num_heads=self.n_head, same_key_value=same_key_value)


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def get_G_prompt(self,batch_size):
        if self.use_embed_for_G_prompt:
            g_prompt_tokens = self.g_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
            g_prompts = self.prefix_encoder(g_prompt_tokens)
        else:
            g_prompts = self.one_g_prompts.expand(batch_size, self.g_prompt_length,768).to(self.roberta.device)
        return g_prompts

    def get_T_prompt(self,batch_size):
        pass

    def get_S_prompt(self,batch_size):
        s_prompt_tokens = self.s_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        s_prompts = self.prefix_encoder(s_prompt_tokens)
        return s_prompts

    def get_M_prompt(self,batch_size):
        pass



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            type_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )


        if self.use_G_prompt:
            g_prompts = self.get_G_prompt(batch_size=batch_size)
        else:
            g_prompts = None

        if train:
            start = task_id * self.s_prompt.s_top_k
            end = (task_id + 1) * self.s_prompt.s_top_k
            single_prompt_mask = torch.arange(start, end).to(raw_embedding.device)
            s_prompt_mask = single_prompt_mask.unsqueeze(0).expand(raw_embedding.shape[0], -1)
            if end > self.s_prompt.s_pool_size:
                s_prompt_mask = None
            t_prompt_mask = torch.arange(type_id,type_id+1).unsqueeze(0).expand(raw_embedding.shape[0], -1)

        else:
            s_prompt_mask = None
            t_prompt_mask = None
        res = self.s_prompt(raw_embedding, s_prompt_mask=s_prompt_mask,t_prompt_mask=t_prompt_mask, cls_features=cls_features)
        s_prompts = res['s_batched_prompt'].squeeze()
        t_prompts = res['t_batched_prompt'].squeeze()
        m_prompts = res['m_batched_prompt'].squeeze()

        if self.use_G_prompt:
            inputs_embeds = torch.cat((g_prompts,t_prompts,s_prompts, m_prompts,raw_embedding), dim=1)
        else:
            inputs_embeds = torch.cat((t_prompts,s_prompts,m_prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.g_prompt_length+self.t_prompt_length+self.s_prompt_length+self.m_prompt_length*self.m_top_k).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLogAD_L2P(nn.Module):
    def __init__(self, model_path,num_labels=2,hidden_dropout_prob=0.1,
        use_prompt_mask = False, embedding_key = 'cls', prompt_init = 'uniform',
        batchwise_prompt = False, prompt_key_init = 'uniform', head_type = 'token',
        s_prompt_length = None,
        s_prompt_pool = False, s_prompt_key = False, s_pool_size = None,
        s_top_k = None,
        s_prompt_layer_idx = None, use_prefix_tune_for_s_prompt = True, same_key_value = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.s_prefix_length = s_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads




        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.s_top_k = s_top_k


        self.s_prompt = SPrompt(length=s_prompt_length, embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, pool_size=s_pool_size,
                                top_k=s_top_k,
                                batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                use_prefix_tune_for_e_prompt=True,
                                num_heads=self.n_head, same_key_value=same_key_value)


        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size,s_past_key_values):

        # past_key_values = torch.cat((s_past_key_values),dim=3)
        past_key_values = s_past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            start = task_id * self.s_prompt.top_k
            end = (task_id + 1) * self.s_prompt.top_k
            single_prompt_mask = torch.arange(start, end).to(input_ids.device)
            s_prompt_mask = single_prompt_mask.unsqueeze(0).expand(input_ids.shape[0], -1)
            if end > self.s_prompt.pool_size:
                s_prompt_mask = None
        else:
            s_prompt_mask = None

        res = self.s_prompt(None, prompt_mask=s_prompt_mask,cls_features=cls_features)
        s_past_key_values = res['batched_prompt'].squeeze()

        past_key_values = self.get_G_prompt(batch_size=batch_size,s_past_key_values=s_past_key_values,)

        prefix_attention_mask = torch.ones(batch_size, self.s_prefix_length*self.s_top_k).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForLogAD_Meanprompt(nn.Module):
    def __init__(self, model_path,num_labels=2,hidden_dropout_prob=0.1,
        use_prompt_mask = False, embedding_key = 'cls', prompt_init = 'uniform',
        batchwise_prompt = False, prompt_key_init = 'uniform', head_type = 'token',
        g_prompt_length = None, s_prompt_length = None, t_prompt_length=None,
        s_prompt_pool = False, s_prompt_key = False, s_pool_size = None,
        s_top_k = None, t_pool_size=None,t_top_k=None, m_prompt_length=None, m_pool_size=None, m_top_k=None,
        s_prompt_layer_idx = None, use_prefix_tune_for_s_prompt = True, same_key_value = False,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, self.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.g_prefix_length = g_prompt_length
        self.s_prefix_length = s_prompt_length
        self.t_prefix_length = t_prompt_length
        self.m_prefix_length = m_prompt_length
        self.n_layer = self.roberta.config.num_hidden_layers
        self.n_head = self.roberta.config.num_attention_heads
        self.n_embd = self.roberta.config.hidden_size // self.roberta.config.num_attention_heads

        # self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        # self.prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.roberta.config.hidden_size, 512,
        #                                                  self.n_layer)


        g_prompt_shape = (self.g_prefix_length, self.n_layer * 2, self.n_head, self.n_embd)
        self.g_prefix = nn.Parameter(torch.randn(g_prompt_shape))


        self.s_prompt_pool = s_prompt_pool
        self.s_prompt_key = s_prompt_key
        self.s_pool_size = s_pool_size
        self.t_pool_size = t_pool_size
        self.m_pool_size = m_pool_size
        self.s_top_k = s_top_k
        self.t_top_k = t_top_k
        self.m_top_k = m_top_k

        self.s_prompt = SPrompt_mean(s_length=s_prompt_length, t_length=t_prompt_length,m_length=m_prompt_length,embed_dim=self.roberta.config.hidden_size,
                                embedding_key=embedding_key, prompt_init=prompt_init,
                                prompt_pool=s_prompt_pool, prompt_key=s_prompt_key, s_pool_size=s_pool_size,
                                s_top_k=s_top_k, t_pool_size=t_pool_size,t_top_k=t_top_k,
                                m_pool_size=m_pool_size,m_top_k=m_top_k,
                                batchwise_prompt=batchwise_prompt,
                                prompt_key_init=prompt_key_init, num_layers=self.n_layer,
                                use_prefix_tune_for_e_prompt=use_prefix_tune_for_s_prompt,
                                num_heads=self.n_head, same_key_value=same_key_value)


        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_G_prompt(self, batch_size,s_past_key_values,t_past_key_values,m_past_key_values):
        g_past_key_values = self.g_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).to(self.roberta.device)
        # g_past_key_values = self.dropout(g_past_key_values)
        g_past_key_values = g_past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = torch.cat((g_past_key_values,t_past_key_values,s_past_key_values,m_past_key_values),dim=3)
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_id=-1,
            type_id=-1,
            cls_features=None,
            train=False,
    ):
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        batch_size = input_ids.shape[0]

        if train:
            start = task_id * self.s_prompt.s_top_k
            end = (task_id + 1) * self.s_prompt.s_top_k
            single_prompt_mask = torch.arange(start, end).to(input_ids.device)
            s_prompt_mask = single_prompt_mask.unsqueeze(0).expand(input_ids.shape[0], -1)
            if end > self.s_prompt.s_pool_size:
                s_prompt_mask = None
            t_prompt_mask = torch.arange(type_id, type_id + 1).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            s_prompt_mask = None
            t_prompt_mask = None
        res = self.s_prompt(None, s_prompt_mask=s_prompt_mask, t_prompt_mask=t_prompt_mask,cls_features=cls_features)
        s_past_key_values = res['s_batched_prompt'].squeeze()
        t_past_key_values = res['t_batched_prompt'].squeeze()
        m_past_key_values = res['m_batched_prompt'].squeeze()
        if train==False:
            s_prompt_idx = res['s_prompt_idx']
            t_prompt_idx = res['t_prompt_idx']

        else:
            s_prompt_idx = None
            t_prompt_idx =None

        past_key_values = self.get_G_prompt(batch_size=batch_size,s_past_key_values=s_past_key_values,
                                            t_past_key_values=t_past_key_values,m_past_key_values=m_past_key_values)


        prefix_attention_mask = torch.ones(batch_size, self.g_prefix_length+self.t_prefix_length*self.t_top_k+self.s_prefix_length*self.s_top_k+self.m_prefix_length*self.m_top_k).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        if train:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )