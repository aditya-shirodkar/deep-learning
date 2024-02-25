import tiktoken
import torch
import torch.nn as nn

torch.manual_seed(1337)

# load file
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# tokenise - try out SentencePiece by Google, tiktoken by OpenAI
print("Characters in dataset: ", len(text))
print(text[:1000])
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(
    "All characters in input file (includes newline char): \n{}".format("".join(chars))
)
print(
    "Number of tokens i.e. vocabulary (simple per-character tokenisation): ", vocab_size
)

str_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_str = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [str_to_int[c] for c in x]
decode = lambda x: "".join([int_to_str[i] for i in x])

phrase = "Hello world!"
print("\nSimple per-character encoder for phrase '{}'".format(phrase))
print(encode(phrase))
print(decode(encode(phrase)))
print("\n")

# tokenising with tiktoken
encoding_tiktoken = tiktoken.get_encoding("gpt2")
print(
    "\nNumber of tokens i.e. vocabulary (tiktoken gp2 style): ",
    encoding_tiktoken.n_vocab,
)
print("tiktoken gp2 style encoder for phrase '{}'".format(phrase))
print(encoding_tiktoken.encode(phrase))
print(encoding_tiktoken.decode(encoding_tiktoken.encode(phrase)))

# store dataset as PyTorch tensor after encoding
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# train-validation splits
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

context_size = 8
print("context size = ", context_size)
# as the target is offset by one index to the right, we need to process 9 elements for context_size = 8
print(train_data[: context_size + 1])
x = train_data[
    :context_size
]  # the training data is not the element at an index but all elements up till the index
y = train_data[
    1 : context_size + 1
]  # next tensor of length 8, each being the target to the input at the same index

# to visualise the above:
for t in range(context_size):
    context = x[: t + 1]
    target = y[t]
    print("when input is {}, the target is {}".format(context.tolist(), target))


batch_size = 4  # number of context_size-long sequences processed at a time in parallel


def get_batch(split):
    # generate a batch of data as inputs x and targets y
    d = train_data if split == "train" else val_data
    # randomly selecting batch_size-long tensors from the training/validation d to sample from
    ix = torch.randint(len(d) - context_size, (batch_size,))
    x = torch.stack([d[i : i + context_size] for i in ix])
    y = torch.stack([d[i + 1 : i + context_size + 1] for i in ix])
    return x, y


x_batch, y_batch = get_batch("train")
print("\ninputs:")
print(x_batch.shape)
print(x_batch)
print("\ntargets:")
print(y_batch.shape)
print(y_batch)

# visualise this parallel batch-processing
for t in range(context_size):  # time dimension in the parallel process
    print("\ntime = ", t)
    for b in range(batch_size):  # batch dimension (which batch?)
        context = x_batch[b, : t + 1]
        target = y_batch[b, t]
        print(
            "batch {}: when input is {}, the target is {}".format(
                b, context.tolist(), target
            )
        )


class BigramLanguageModel(nn.Module):
    def __init__(self, channel_size):  # channel_size = vocab_size of the data
        super().__init__()
        self.token_embedding_table = nn.Embedding(channel_size, channel_size)

    # forward-pass required function for nn.Module; targets=None to accommodate use in the generate function below
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(
            idx
        )  # (B, T, C) i.e. (batch x time x channel)-sized tensor

        if targets == None:
            loss = None
        else:
            # reforming the above as cross_entropy wants logits as ((B, T), C) 2-D array, not a 3D tensor
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        # logits returned as (B, T, C) if targets=None, else returned as ((B, T), C)
        return logits, loss

    # gives the (B, T) array for the current context at each point of time
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # logits array is (B, T, C); loss unused here
            # the (complete) logits being processed are only those on the last time step (i.e. full context-sized array)
            # this is because it is a bigram model! this makes the concatenation phase below meaningless
            logits = logits[:, -1, :]  # array is now (B, C)
            # softmax to normalise logit outputs to a probability distribution, similarly using the final time step
            probs = nn.functional.softmax(
                logits, dim=-1
            )  # probability of logits array is also (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # concatenate index to already sampled sequence of indices
            # this isn't really required for the bigram model as we're only looking at the previous letter!
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
logits, loss = model(x_batch, y_batch)
print(
    "\nlogits shape : ((batch_dimension x time_dimension), channel_dimension) = (({} x {}), {}) = {}".format(
        batch_size, context_size, vocab_size, logits.shape
    )
)
# loss is the negative log likelihood, -ln(1/vocab_size) equalling around -ln(1/65) in this case
# the dataset being diffuse likely means this estimate isn't very accurate
print("loss: ", loss)

# the first thing we feed is a newline sequence (chars[0] = \n)
# 100 tokens will be generated at a time

print(
    "\nOutputs of an optimised bigram model when initially fed the index of a newline character:"
)
print(
    decode(
        model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[
            0
        ].tolist()
    )
)

# create PyTorch optimiser to reduce loss; lr = learning rate
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
print("\nRunning optimiser")

batch_size = 32
n_runs = 10000
for steps in range(n_runs):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimiser.zero_grad(
        set_to_none=True
    )  # refresh gradients to zero at every iteration
    loss.backward()
    optimiser.step()

    if steps == 0:
        print("Loss after one run of the optimiser: ", loss.item())

print("Loss after {} runs of the optimiser: {}".format(n_runs, loss.item()))

print(
    "\nOutputs of an untrained bigram model (only vocabulary known) when initially fed the index of a \\n character:"
)
print(
    decode(
        model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[
            0
        ].tolist()
    )
)

# mathematical trick in self-attention
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# each of the 8 tokens per batch do not yet communicate with each other
# tokens in the nth location (of 8) must communicate only with the tokens before 1, ..., (n-1)
# i.e. we do not look for information in the future, rather we try to predict it
# VERSION 1: the simplest way to communicate with past tokens is to average the channels of the present and past tokens
# the longer the chain of the past, the less "weight" do we give to each element of that chain.
# when starting with zeroes (no gradient), the weights are equal for every element of the chain.

x_bow = torch.zeros((B, T, C))  # bow = bag-of-words; common terminology
for b in range(B):
    for t in range(T):
        x_prev = x[
            b, : t + 1
        ]  # everything in the current batch,upto and including the current token; has (t, C) dimensions
        x_bow[b, t] = torch.mean(x_prev, 0)

print("\nThe tensor at the zeroth index:")
print(x[0])
print("\nThe tensor after bag-of-words averaging:")
print(x_bow[0])
print(
    "Notice how every element is the average of the elements vertically above it, including itself."
)

# VERSION 2: rather than using loops, we could make this efficient using matrix multiplication. For example:
a1 = torch.ones(3, 3)
b = torch.randint(0, 10, (3, 2)).float()
c1 = a1 @ b  # matrix multiplication

print("\nMatrix multiplication to calculate bag-of-words average, an example:")
print("\na = ", a1)
print("b = ", b)
print("c = a @ b = ", c1)

a2 = torch.tril(torch.ones(3, 3))
c2 = a2 @ b
print("a2 = ", a2)
print("c2 = a2 @ b = ", c2)

a3 = a2 / torch.sum(
    a2, 1, keepdim=True
)  # creates a matrix where every row adds up to one and follows the shape of a2
c3 = a3 @ b
print("a3 = ", a3)
print("c3 = a3 @ b = ", c3)
print(
    "\nc3 here has elements which each are an average of itself and the elements above it."
)

# returning to our problem
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)
print("\nReturning to our problem, setting weights as ")
print(weights)

x_bow2 = (
    weights @ x
)  # (T, T) @ (B, T, C) => PyTorch converts the former to (B, T, T), therefore (B, T, T) @ (B, T, C) => (B, T, C)
print(
    "\nMultiplying the weights with x gives the same matrices as before (almost, it depends on how mean is implemented)"
)
print("\nx_bow:")
print(x_bow)
print("\nx_bow2:")
print(x_bow2)

# VERSION 3: using Softmax
tril = torch.tril(torch.ones(T, T))

print("\nTrilled array of ones:")
print(tril)
weights = torch.zeros((T, T))
print("\nWeights as a zero matrix:")
weights = weights.masked_fill(
    tril == 0, float("-inf")
)  # set tokens in the future to -inf, indicating they aren't to be considered as they cannot communicate with the past
print(
    "\nWeights after filling all zeroes in the trilled array with negative infinities:"
)
print(weights)
weights = nn.functional.softmax(weights, dim=-1)
print("\nWeights after softmaxing; obtains the same weights in the previous version.")
print(
    "\nSoftmaxing works by raising 1 by each element of the array, and then dividing by the sum of each row."
)
print(weights)

x_bow3 = weights @ x
print(
    "\nOnce again, this is another way to average the channels of the tokens until and including the present index."
)
print("x_bow3:")
print(x_bow3)

# VERSION 4: with self-attention
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# no longer want initial weights to be uniform each time
# i.e., each previous token may not have been as important to the current token/have an affinity for it
# every single token at each position emits two vectors, a query and a key
# each token now queries for a key with which it has a greater affinity
# single head performing self-attention:
head_size = 16
key = nn.Linear(C, head_size, bias=False)  # what do I contain?
query = nn.Linear(C, head_size, bias=False)  # what am I looking for?
value = nn.Linear(
    C, head_size, bias=False
)  # rather than x itself, we only use data that fits within one head_size
k = key(x)  # (B, T, head_size)
q = query(x)  # (B, T, head_size)
v = value(x)  # (B, T, head_size)
# need to dot product the query and the key, so we transpose the last two columns of the key matrix (k)
# this is done to compute the attention scores/affinities (usefulness of another token/node to the current)
weights = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) => (B, T, T)

tril = torch.tril(torch.ones(T, T))
weights = weights.masked_fill(tril == 0, float("-inf"))
weights = nn.functional.softmax(weights, dim=-1)
print("\nWeights when a single head is performing self-attention:")
print(weights)
print(
    "Weights are different as the latest token of each batch is querying for keys for each previous token."
)

x_bow4 = weights @ v
print(
    f"\nBag-of-words averaging with self-attention implemented for a single head of size {head_size}:"
)
print("x_bow4:")
print(x_bow4)

# Notes on attention:
# 1) It's a communication mechanism as a series of nodes pointing to themselves and all nodes before them.
# 2) There isn't a space dimension; therefore we need to create it by positionally encoding them while tokenising.
# 3) Elements across batches do not communicate.
# 4) The concept of future nodes not talking to each other is specific to the case of decoder blocks, like GPT creation
# for something like sentiment analysis (encoder blocks), everything communicates with each other.
# 5) Self-attention means that values, keys, and queries all come from the same source. Cross-attention is when queries
# come from one source but values and keys from another.
# 6) for Q = query, K = key, V = value, T = time, d_k = dimension/head size (same as key size)
# attention(Q, K, V) = softmax(QK^T/(d_k)^-1)*V
# dividing by the root of the head size is called scaled attention (sets variance to 1). This is done to keep the
# initial weights diffuse (i.e. not too extreme). If they're not diffuse, softmax might converge to one-hot vectors.
# One-hot vectors are those with a single high coefficient and all others low. Having such a situation would mean a
# node (token) may only converse with a single other rather than attaining information from all nodes.

# MULTI-HEADED ATTENTION
# multiple attention heads in parallel

# FEED-FORWARD NETWORK
# basically a multi-layer perceptron (MLP)
# FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
# feed-forward is done independently by each token; once they've gathered information from self-attention, the tokens
# "disconnect" and think on what they've learned

# Generally, we want to mix the communication and computation layers, by using blocks of codes which process multiple
# nodes in parallel. However, this runs into the issue that deep neural nets face w.r.t. optimisation issues.
# To resolve these issues, two optimisations can help:
# 1) Skip-connections: Transform the data, but also pass the older data by addition.
# Addition distributes gradients equally to both branches that feed as the input (the transformed and untransformed).
# Therefore, supervision (gradients from the loss) propagate through every block.
# 2) Layernorm: Addition is not sufficient; normalising these addition results is also very important.
# Layer norm is similar to batch normalisation. Batch normalisation ensures that, across the batch dimension (columns),
# every individual neuron has unit Gaussian distribution i.e.[-1, 1] std deviation. Layer normalisation instead
# normalises across rows rather than columns - no buffers are needed as we aren't working across batch and there's no
# difference between training and testing. Nowadays, layernorm is applied BEFORE the transformation (as opposed to
# doing it simultaneously with addition, as was in the "Attention is all you need" paper). This is called the
# "pre-norm formulation."

# SCALING UP THE MODEL
# So far, the architecture of our transformer is decoder-only; i.e. it cannot take user inputs but rather just considers
# the training data and generates text from there. Inputs could be: questions asked, text to translate, multimedia
# inputs like images, etc. Different encodings are needed in each case. The encoder does not use a triangular mask
# (the tril masked fill), and instead allows tokens to talk as much as they want, as there's no concept of time.
# Cross-attention is then used to communicate with the main transformer (the keys and values come from the encoder
# block. Therefore, it conditions the decoder not only on the past of the decoder's training, but also on the fully
# encoded user-input prompt.
