import math

import torch
from torch import nn, optim
from torch.nn import Transformer
from unit.conftest import DEVICE

from torchjd.aggregation import UPGradWeighting
from torchjd.autogram import Engine


# ----- 2. Simple Transformer model -----
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size=32, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size)
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, src, tgt):
        # Shape: [batch_size, seq_len]
        src_emb = self.pos_encoder(self.embedding(src))  # [batch_size, seq_len, emb_size]
        tgt_emb = self.pos_encoder(self.embedding(tgt))  # [batch_size, seq_len, emb_size]
        output = self.transformer(src_emb, tgt_emb)  # [batch_size, seq_len, emb_size]
        return self.fc_out(output)  # [batch_size, seq_len, vocab_size]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, x.size(1), :]


def main():
    sentences = [
        "i love to code in python lol",
        "pytorch is not an easy framework haha",
        "i love transformers and convolutional networks mdr",
        "transformers are powerful and very deep mdr",
        "deep learning is something that exists lol",
    ]

    vocab = sorted(set(" ".join(sentences).split()))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)

    def encode(sentence):
        return torch.tensor([word_to_ix[w] for w in sentence.split()], dtype=torch.long)

    encoded_sentences = [encode(s) for s in sentences]

    model = SimpleTransformerModel(vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    engine = Engine(model, batch_dim=0)
    W = UPGradWeighting()

    # ----- 4. Training loop -----
    for epoch in range(200):
        total_loss = 0

        srcs = []
        tgt_inputs = []
        tgt_outputs = []
        for seq in encoded_sentences:
            src_i = seq[:-1]
            tgt_input_i = seq[:-1]
            tgt_output_i = seq[1:]

            srcs.append(src_i)
            tgt_inputs.append(tgt_input_i)
            tgt_outputs.append(tgt_output_i)

        src = torch.stack(srcs, dim=0)
        tgt_input = torch.stack(tgt_inputs, dim=0)
        tgt_output = torch.stack(tgt_outputs, dim=0)

        optimizer.zero_grad()
        output = model(src, tgt_input).movedim([2], [1])

        losses = criterion(output, tgt_output)
        loss = losses.mean()

        # Autograd
        # loss.backward()

        # Autogram
        loss_vector = losses.mean(dim=1)
        gramian = engine.compute_gramian(loss_vector)
        weights = W(gramian)
        print(weights)
        loss_vector.backward(weights)

        optimizer.step()
        total_loss += loss.mean().item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    main()
