"""
Edit hyperparameters ONLY HERE.
This dict is loaded by the experiment builder and merged into dataclass configs.
"""

CONFIG = dict(
    # ── Model ──────────────────────────────────────────────
    vocab_size=50257,
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=256,
    dropout=0.2,
    bias=False,

    # ── Data ───────────────────────────────────────────────
    dataset="shakespeare",  # "shakespeare" or path to .txt
    batch_size=64,

    # ── Training ───────────────────────────────────────────
    max_steps=5000,
    learning_rate=3e-4,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    warmup_steps=100,
    lr_decay_steps=5000,
    min_lr=3e-5,

    # ── Evaluation / Logging ───────────────────────────────
    eval_interval=250,
    eval_steps=20,
    log_interval=10,

    # ── Checkpointing ─────────────────────────────────────
    checkpoint_dir="checkpoints",
    save_interval=1000,

    # ── Visualisation ─────────────────────────────────────
    plot=True,
    plot_interval=50,

    # ── Sampling ──────────────────────────────────────────
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
)

