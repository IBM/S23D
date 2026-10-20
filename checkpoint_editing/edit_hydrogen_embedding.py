#!/usr/bin/env python

import argparse

import torch


def edit_hydrogen_embedding(
    input_filename, output_filename, hydrogen_index=9, mask_token_index=1
):
    content = torch.load(
        input_filename, map_location=torch.device("cpu"), weights_only=False
    )
    state_dict = content["state_dict"]
    embedding = state_dict["score_network.graph_encoder.embedding.weight"]
    embedding[hydrogen_index] = embedding[mask_token_index]
    torch.save(content, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edit existing S23D checkpoint file to incorporate Hs."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file to edit",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the output checkpoint file. Default: <checkpoint>_h.ckpt",
    )
    args = parser.parse_args()

    checkpoint_name = args.checkpoint
    output_checkpoint_name = (
        args.output or f"{args.checkpoint.removesuffix('.ckpt')}_h.ckpt"
    )
    edit_hydrogen_embedding(checkpoint_name, output_checkpoint_name)
