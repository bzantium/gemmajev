"""Send a JSON request to a trained GemmaJev model and print its probabilities."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "mlx", "transformers", "vllm"), required=True)
    parser.add_argument(
        "--model", required=True, help="Training run (JAX) or exported model directory"
    )
    parser.add_argument("--request", default="examples/maze_request.json")
    parser.add_argument("--device", default="cpu", help="Transformers device, e.g. cpu or cuda")
    parser.add_argument("--threads", type=int, default=8, help="Transformers CPU thread count")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM server URL")
    parser.add_argument("--served-model", default="gemmajev", help="vLLM served model name")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.backend == "vllm":
        import os

        from gemmajev.vllm_backend import VLLMGameEngine

        engine = VLLMGameEngine(
            args.model,
            base_url=args.base_url,
            model=args.served_model,
            api_key=os.environ.get("VLLM_API_KEY"),
        )
    elif args.backend == "mlx":
        from gemmajev.mlx_backend import MLXGameEngine

        engine = MLXGameEngine(args.model)
    elif args.backend == "transformers":
        import torch

        from gemmajev.transformers_backend import TransformersGameEngine

        torch.set_num_threads(args.threads)
        engine = TransformersGameEngine(args.model, device=args.device)
    else:
        from gemmajev.jax_backend import GameEngine

        engine = GameEngine(args.model, compact=True)
    request = json.loads(Path(args.request).read_text())
    print(json.dumps(engine.predict(request), indent=2))


if __name__ == "__main__":
    main()
