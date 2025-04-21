import torch


BEGINNING_TOKEN = 127 # The beginning of file token to start generating. We use this value since it's unused
END_TOKEN = 0 # The end of file token to stop generating


def attention_mask(seq_length: int, learned_embedding: torch.Tensor) -> torch.Tensor:
    """
    Create an attention mask.
    :param seq_length: The length of the sequence.
    :param learned_embedding: The learned embedding.
    Format is [embed_0, embed_1, ..., embed_n], meaning 'embed_n' is n positions from our target 'x'.
    :return: The attention mask.
    """
    # learned_embedding: [embed_0, embed_1, ..., embed_n]
    embed = learned_embedding.new_full((*learned_embedding.shape[:-1], seq_length, seq_length), float('-inf')) # Create the embed matrix and fill it with negative infinite
    pos = torch.arange(seq_length)
    rel_pos = pos[:, None] - pos[None, :]
    valid_pos = (0 <= rel_pos) & (rel_pos < learned_embedding.shape[-1])

    embed[..., valid_pos] = learned_embedding[..., rel_pos[valid_pos]]
    return embed


class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, res_pos_length: int = 512):
        super().__init__()
        self._res_pos = torch.nn.Parameter(torch.zeros(num_heads, res_pos_length))
        self._self_att = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embed_dim, embed_dim),
        )
        self._in_norm = torch.nn.LayerNorm(embed_dim)
        self._mlp_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self._in_norm(x)
        attn_mask = attention_mask(x.shape[1], self._res_pos)
        x = x + self._self_att(x_norm, x_norm, x_norm, attn_mask=attn_mask)[0] # Get the results of the attention layer. We don't want the weights. That's why we have the index
        x = x + self._mlp(self._mlp_norm(x))
        return x

class Transformer(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        num_embeddings = 128
        self._network = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim), # Embed the input before feeding it to the transformer
            *[TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)],
            torch.nn.Linear(in_features=embed_dim, out_features=num_embeddings) # Classify each output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network(x)

def train():
    with open(__file__) as f:
        code = f.read()
    tokens = torch.as_tensor([BEGINNING_TOKEN] + [ord(c) for c in code] + [END_TOKEN]) # Convert the characters to their Unicode codes (integers)

    net = Transformer(embed_dim=256, num_heads=8, num_layers=4)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)

    if torch.cuda.is_available():
        net = net.cuda()
        tokens = tokens.cuda()

    for i in range(400):
        pred = net(tokens[None, :-1])[0] # Predict for every token in our sequence, except for the last one as it will be the end of sentence token
        loss = torch.nn.functional.cross_entropy(pred, tokens[1:]) # Predict on the next character. We don't predict on the first token
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 10 == 0:
            print(f'Iteration {i}: {float(loss)}')
            pred = net(tokens[None, :10])[0]
            print(tokens[1:11])
            print(pred.argmax(-1))
            print([int(net(tokens[None, :n+1])[0,-1].argmax()) for n in range(10)])

    net.eval()
    net.cpu() # Move the model to the CPU
    torch.save(net, "transformer.pth")


def sample():
    import sys

    net = torch.load("transformer.pth")
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    data = [127]
    for i in range(10000):
        tokens = torch.as_tensor(data[-500:])
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        pred = net(tokens[None])[0, -1]
        next_char = pred.argmax(-1)

        if next_char == END_TOKEN:
            # The model predicted the end of file, stop predicting
            break

        data.append(int(next_char))
        sys.stdout.write(chr(int(next_char)))
        sys.stdout.flush()


if __name__ == "__main__":
    train()
    sample()