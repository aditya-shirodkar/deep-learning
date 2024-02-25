import torch
import torch.nn as nn

# hyperparameters
batch_size = (
    64  # number of batches of inputs of length block_size processed in parallel
)
block_size = 256  # context window size
max_iters = 5000  # model optimisation iterations
eval_interval = 500  # evaluation intervals over which mean loss is calculated
learning_rate = 3e-4  # optimisation model learning rate
device = "cuda" if torch.cuda.is_available() else "cpu"  # load GPU if available
eval_iters = 200
n_embed = 384  # number of embeddings in the token embedding table
n_head = 6  # number of heads; n_embed / n_head = batch_size
n_layer = 6  # number of layers of processing
dropout = 0.2  # proportion of nodes "disconnected" from others; prevents over-communication and hence over-fitting

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mappings for encoders/decoders
str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [str_to_int[c] for c in x]
decode = lambda x: "".join([int_to_str[i] for i in x])

# train-test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # subtract context size to avoid index out of bounds
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)  # after loading data, move to device
    return x, y


# single head of self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # creating a buffer for a tril matrix (lower triangular ones)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5  # scaled attention
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # future nodes blocked (decoder block)
        weights = nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # self-attention results
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # projection back into the residual pathway, which is linear (skip-connection)
        out = self.dropout(self.proj(out))
        return out


# without feed-forward, communication attained via self-attention isn't as effective, as
# tokens do not have time to think about what they found from the other tokens
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # multiplying by 4 because the inner layer has 4 x the dimensionality of the model
        # the inner layer is larger, therefore, than the outer and can perform more computation (?)
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),  # rectified linear unit; solves vanishing gradients issue by introducing non-linearity
            nn.Linear(
                4 * n_embed, n_embed
            ),  # projection layer; back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# communication + computation together
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(head_size)
        self.ffwd = FeedForward()
        # layer norms; ln1 for the self-attention heads and ln2 for the feed-forward
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # addition is done here to provide skip-connections, thereby preserving pre-transformation gradients
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define token and position embedding tables; the latter doesn't matter when there is no self-attention
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # self.blocks = nn.Sequential(
        #     Block(), Block(), Block(), nn.LayerNorm(n_embed)
        # )
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)  # final layernorm
        # self.sa_head = Head(n_embed)  # single self-attention head
        self.sa_heads = MultiHeadAttention(
            4
        )  # multiple self-attention heads in parallel => similar to group-convolutions
        self.ffwd = FeedForward()  # feed-forward network
        self.lm_head = nn.Linear(n_embed, vocab_size)  # language-modelling head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = token_embeddings + pos_embeddings  # (B, T, C)
        # x = self.sa_head(x)  # apply one head of self-attention
        x = self.sa_heads(x)  # apply multiple heads of self-attention; (B, T, C)
        x = self.ffwd(x)  # apply feed-forward; (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size); vocab size ~= C

        if targets == None:
            loss = None
        else:
            n_batch, n_time, n_channel = logits.shape
            logits = logits.view(n_batch * n_time, n_channel)
            targets = targets.view(n_batch * n_time)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # to prevent idx being out of block size range now that positional embedding is used
            idx_condensed = idx[:, -block_size:]
            # get logits
            logits, loss = self(idx_condensed)
            # the following makes the model bigrammatic by using only the previous time step:
            logits = logits[:, -1, :]
            # generate probabilities with softmax
            probs = nn.functional.softmax(logits, dim=-1)  # (B, C)
            # sample from the multinomial distribution using these probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sample to previously generated
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)  # move model to device


@torch.no_grad()  # PyTorch will not call .backward() on this as backpropagation isn't needed
def estimate_loss():
    # average out losses over eval_iters number of iterations to smoothen the noise in the decreasing loss function
    out = {}
    model.eval()  # transition model to evaluation phase
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # revert model to training phase
    # model phase transitions don't matter for this simple model, train and eval phases do the same thing
    return out


optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # only calculating loss every eval_interval to smoothen decreasing loss function
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}"
        )
        # when train loss < val loss, it means there's some over-fitting; can be lowered by using dropout

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True)  # reset loss gradient to zero every iteration
    loss.backward()
    optimiser.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # move context to device
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
