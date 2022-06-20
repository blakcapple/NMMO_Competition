import torch.nn as nn 

class SelfAtten(nn.Module):
    '''self attention layer'''
    def __init__(self, input_dim, query_dim, embed_dim, atten_head):
        super().__init__()
        self.atten_input_dims = input_dim
        self.query_dims = query_dim
        self.value_dims = query_dim
        self.key_dims = query_dim
        self.embed_dim = embed_dim
        self.atten_head = atten_head
        self.query = nn.Linear(self.atten_input_dims, self.query_dims)
        self.value = nn.Linear(self.atten_input_dims, self.value_dims)
        self.key = nn.Linear(self.atten_input_dims, self.key_dims)
        self.multiattention = nn.MultiheadAttention(self.embed_dim, self.atten_head, batch_first=True)
        self.act = nn.ReLU()
    
    def forward(self,input, mask=None):
        proj_query = self.query(input)
        proj_key = self.key(input)
        proj_value = self.value(input)
        output, _ = self.multiattention(proj_query, proj_key, proj_value, mask)
        
        return self.act(output)