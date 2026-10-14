# archviz — model architecture from metadata alone

Turns a Hugging Face repo id into an accurate data-flow diagram of the model's
computation graph, using only the files a repo publishes: `config.json`,
`generation_config.json`, `preprocessor_config.json`, and friends.

## Why not just download the weights?

Because the architecture is *declared*, not *learned*. Every dimension a diagram
needs — layer counts, widths, head counts, the vocabulary size — is written in
`config.json`. Fetching 16 GB of weights to read a few hundred numbers is slow,
expensive, and pointless; and a diagram built from a truncated download is
usually wrong in ways nobody notices.

## Honesty rules

- Nothing is `measured`. Shapes are `derived` from config, parameter counts are
  computed from those shapes, and the diagram says which is which.
- A dimension that is absent is not guessed. The node says what is missing.
- Repetition is drawn as `× N` with one expandable instance, never as N boxes.

## Usage

    python archviz/graph.py Qwen/Qwen3-4B --json          # graph as JSON
    python archviz/graph.py Qwen/Qwen3-4B --md           # a table
