"""
Entry point that simply proxies to ``data_extraction.dump_activations``.
"""

from data_extraction import main as dump_activations_cli


if __name__ == "__main__":
    dump_activations_cli()
