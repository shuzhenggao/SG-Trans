# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py
""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
#from c2nl.utils.misc import generate_relative_positions_matrix, relative_matmul

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, d_k, d_v, dropout=0.1,
                 max_relative_positions=0, use_neg_dist=True, coverage=False):
        super(MultiHeadedAttention, self).__init__()

        self.head_count = head_count
        self.model_dim = model_dim
        self.d_k = d_k
        self.d_v = d_v

        self.key = nn.Linear(model_dim, head_count * self.d_k)
        self.query = nn.Linear(model_dim, head_count * self.d_k)
        self.value = nn.Linear(model_dim, head_count * self.d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.head_count * d_v, model_dim)
        self._coverage = coverage

        #self.max_relative_positions = max_relative_positions
        self.use_neg_dist = use_neg_dist

        '''if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1 \
                if self.use_neg_dist else max_relative_positions + 1
            self.relative_positions_embeddings_k = nn.Embedding(
                vocab_size, self.d_k)
            self.relative_positions_embeddings_v = nn.Embedding(
                vocab_size, self.d_v)'''

    def forward(self, key, value, query,code_keyword_rep=None,code_intoken_map=None,code_instatement_map=None,code_dataflow=None,
                heads_type=None,mask=None, layer_cache=None,attn_type=None, step=None, coverage=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        use_gpu = key.is_cuda

        def shape(x, dim):
            """  projection """
            return x.view(batch_size, -1, head_count, dim).transpose(1, 2)

        def unshape(x, dim):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                # 1) Project key, value, and query.
                key = shape(self.key(key), self.d_k)
                value = shape(self.value(value), self.d_v)
                query = shape(self.query(query), self.d_k)

                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value

            elif attn_type == "context":
                query = shape(self.query(query), self.d_k)
                if layer_cache["memory_keys"] is None:
                    key = shape(self.key(key), self.d_k)
                    value = shape(self.value(value), self.d_v)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = shape(self.key(key), self.d_k)
            value = shape(self.value(value), self.d_v)
            query = shape(self.query(query), self.d_k)

        '''if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions, self.use_neg_dist,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings_k(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings_v(
                relative_positions_matrix.to(key.device))'''

        key_len = key.size(2)
        query_len = query.size(2)


        # 2) Calculate and scale scores.
        query = query / math.sqrt(self.d_k)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        scores = query_key
        scores = scores.float()

        if code_intoken_map is not None:
            code_intoken_map = -1e9*code_intoken_map
            code_instatement_map = -1e9*code_instatement_map
            lens = scores.shape[2]
            # print(scores.shape)
            # print(code_instatement_map.shape)
            code_dataflow = code_dataflow[:,0:lens,0:lens].float()
            #code_dataflow = code_dataflow.permute(0,2,1)
            #code_controlflow = code_controlflow[:,0:lens,0:lens]
            zero_map = torch.zeros(scores.shape[0],scores.shape[2],scores.shape[2]).cuda(non_blocking=True)
            local_mask_map = torch.stack([code_intoken_map for i in range(heads_type[0])]+
                                         [code_instatement_map for i in range(heads_type[1])]+
                                         [zero_map for i in range(heads_type[2])]+
                                         [zero_map for i in range(heads_type[3])],1)
            global_enhence_map = torch.stack([zero_map for i in range(heads_type[0])]+
                                             [zero_map for i in range(heads_type[1])]+
                                             [code_dataflow for i in range(heads_type[2])]+
                                             [zero_map for i in range(heads_type[3])],1)
            global_enhence_map = 5*global_enhence_map.mul(scores)
            scores = scores + local_mask_map + global_enhence_map
            

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        
        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        
        '''if code_keyword_rep is not None:
            global_add_map = torch.stack([code_keyword_map for i in range(heads_type[0])]+
                                    [code_keyword_map for i in range(heads_type[1])]+
                                    [zero_map for i in range(heads_type[2])],1)
            scores = attn+global_add_map

            if mask is not None:
                scores = scores.masked_fill(mask, -1e18)
            if heads_type[0] + heads_type[1] != 0:
                attn = self.softmax(scores).to(query.dtype)'''

        # ----------------------------

        # 3) Apply attention dropout and compute context vectors.
        # attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original, self.d_v)

        final_output = self.output(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # a list of size num_heads containing tensors
        # of shape `batch x query_len x key_len`
        attn_per_head = [attn.squeeze(1)
                         for attn in attn.chunk(head_count, dim=1)]

        covrage_vector = None

        return final_output, attn_per_head, covrage_vector

    def update_dropout(self, dropout):
        self.dropout.p = dropout
