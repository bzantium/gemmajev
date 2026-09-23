"""Send a JSON request to a trained GemmaJev model and print its probabilities."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "mlx"), required=True)
    parser.add_argument("--model", required=True, help="Training run (JAX) or exported model (MLX)")
    parser.add_argument("--request", default="examples/maze_request.json")
    args = parser.parse_args()
    if args.backend == "mlx":
        from gemmajev.mlx_backend import MLXGameEngine

        engine = MLXGameEngine(args.model)
    else:
        from gemmajev.jax_backend import GameEngine

        engine = GameEngine(args.model, compact=True)
    request = json.loads(Path(args.request).read_text())
    print(json.dumps(engine.predict(request), indent=2))


if __name__ == "__main__":
    main()
