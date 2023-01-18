max_pow = 5

aquamam_toy = {
    "model_args": {
        "toy_args": {"is_toy": True, "max_pow": max_pow},
        "L": 6,
        "d_model": 64,
        "nhead": 8,
        "dropout": 0.0,
        "num_layers": 3,
        "n_bins": 50257,  # GPT-3 vocaublary size.
    },
    "batch_size": 128,
    "patience": 5,
    "num_workers": 2,
    "lr": 1e-4,
    "epochs": 1000,
    "test_batch_size": 1536,
    "beam_k": 1,
}
aquamam_mog_toy = {
    "model_args": {
        "toy_args": {"is_toy": True, "max_pow": max_pow},
        "L": 6,
        "d_model": 64,
        "nhead": 8,
        "dropout": 0.0,
        "num_layers": 3,
        "n_comps": 512,
    },
    "batch_size": 128,
    "patience": 5,
    "num_workers": 2,
    "lr": 1e-4,
    "epochs": 1000,
    "test_batch_size": 1536,
    "beam_k": 1,
}
aquamam_solid = {
    "model_args": {
        "toy_args": {"is_toy": False},
        "L": 6,
        "d_model": 512,
        "nhead": 8,
        "dropout": 0.1,
        "num_layers": 6,
        "n_bins": 500,
    },
    "batch_size": 128,
    "patience": 5,
    "num_workers": 2,
    "lr": 1e-4,
    "epochs": 1000,
    "test_batch_size": 1024,
    "predict_batch_size": 1536,
    "beam_k": 1,
}
ipdf_toy = {
    "model_args": {
        "toy_args": {
            "is_toy": True,
            "max_pow": max_pow,
            "visual_embedding_size": 2048,
        },
        "resnet": "resnet50",
        "L": 3,
        "n_hidden_nodes": 256,
        "mlp_layers": 4,
    },
    "neg_samples": 4095,
    "batch_size": 128,
    "num_workers": 2,
    "lr": 1e-4,
    "warmup_steps": 1000,
    "iterations": 300000,
    "number_queries": 2000000,
    "test_batch_size": 1,
}
# See: https://github.com/google-research/google-research/tree/master/implicit_pdf#reproducing-symsol-results
# and Section S8.
ipdf_solid = {
    "model_args": {
        "toy_args": {"is_toy": False},
        "resnet": "resnet50",
        "L": 3,
        "n_hidden_nodes": 256,
        "mlp_layers": 4,
    },
    "neg_samples": 4095,
    "batch_size": 128,
    "num_workers": 2,
    "lr": 1e-4,
    "warmup_steps": 1000,
    "iterations": 300000,
    "number_queries": 2000000,
    "test_batch_size": 1,
}
configs = {
    "aquamam": {"toy": aquamam_toy, "cube": aquamam_solid, "cylinder": aquamam_solid},
    "ipdf": {"toy": ipdf_toy, "cube": ipdf_solid, "cylinder": ipdf_solid},
    "aquamam_mog": {"toy": aquamam_mog_toy},
}
