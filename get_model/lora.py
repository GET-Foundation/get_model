import argparse

import torch
from minlora import get_lora_state_dict
from minlora.model import merge_lora


def extract_lora_weights(state_dict):
    """Extract LoRA weights from the state_dict."""
    return get_lora_state_dict(state_dict)


def extract_finetuned_weights(state_dict):
    """Extract finetuned weights from the state_dict (excluding 'original' in key names), this includes both lora keys and other layers that lora was not applied to, including those that was freezed."""
    finetuned_state_dict = {
        name: param for name, param in state_dict.items() if "original" not in name
    }
    return finetuned_state_dict


def load_lora_weights(state_dict, path, model_key="state_dict"):
    """Load LoRA weights from a checkpoint file into the state_dict."""
    lora_state_dict = torch.load(path, weights_only=False)
    state_dict.update(lora_state_dict[model_key])
    return state_dict


def load_finetuned_weights(state_dict, path, model_key="state_dict"):
    """Load finetuned weights from a checkpoint file into the state_dict."""
    finetuned_state_dict = torch.load(path, weights_only=False)
    state_dict.update(finetuned_state_dict[model_key])
    return state_dict


def merge_lora(model: torch.nn.Module):
    """Merge LoRA parametrization to all layers in a model. This will remove all parametrization."""
    return merge_lora(model)


def main():
    parser = argparse.ArgumentParser(
        description="CLI for managing LoRA and finetuned weights"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Extract LoRA weights
    parser_extract_lora = subparsers.add_parser(
        "extract_lora", help="Extract LoRA weights"
    )
    parser_extract_lora.add_argument(
        "input_ckpt", type=str, help="Path to input checkpoint"
    )
    parser_extract_lora.add_argument(
        "output_path", type=str, help="Path to save LoRA weights"
    )

    # Extract finetuned weights
    parser_extract_finetuned = subparsers.add_parser(
        "extract_finetuned", help="Extract finetuned weights"
    )
    parser_extract_finetuned.add_argument(
        "input_ckpt", type=str, help="Path to input checkpoint"
    )
    parser_extract_finetuned.add_argument(
        "output_path", type=str, help="Path to save finetuned weights"
    )

    # Load LoRA weights
    parser_load_lora = subparsers.add_parser(
        "load_lora", help="Load LoRA weights into state dict"
    )
    parser_load_lora.add_argument(
        "input_ckpt", type=str, help="Path to input checkpoint"
    )
    parser_load_lora.add_argument(
        "lora_ckpt", type=str, help="Path to LoRA weights checkpoint"
    )
    parser_load_lora.add_argument(
        "output_path", type=str, help="Path to save updated state dict"
    )
    parser_load_lora.add_argument(
        "--model_key", type=str, default="state_dict", help="Key of model in checkpoint"
    )

    # Load finetuned weights
    parser_load_finetuned = subparsers.add_parser(
        "load_finetuned", help="Load finetuned weights into state dict"
    )
    parser_load_finetuned.add_argument(
        "input_ckpt", type=str, help="Path to input checkpoint"
    )
    parser_load_finetuned.add_argument(
        "finetuned_ckpt", type=str, help="Path to finetuned weights checkpoint"
    )
    parser_load_finetuned.add_argument(
        "output_path", type=str, help="Path to save updated state dict"
    )
    parser_load_finetuned.add_argument(
        "--model_key", type=str, default="state_dict", help="Key of model in checkpoint"
    )

    # Merge LoRA weights
    parser_merge_lora = subparsers.add_parser(
        "merge_lora", help="Merge LoRA weights into model"
    )
    parser_merge_lora.add_argument(
        "model_ckpt", type=str, help="Path to model checkpoint"
    )
    parser_merge_lora.add_argument(
        "output_path", type=str, help="Path to save merged model"
    )

    args = parser.parse_args()

    if args.command == "extract_lora":
        state_dict = torch.load(args.input_ckpt, weights_only=False)["state_dict"]
        lora_weights = extract_lora_weights(state_dict)
        torch.save(lora_weights, args.output_path)
    elif args.command == "extract_finetuned":
        state_dict = torch.load(args.input_ckpt, weights_only=False)["state_dict"]
        finetuned_weights = extract_finetuned_weights(state_dict)
        torch.save(finetuned_weights, args.output_path)
    elif args.command == "load_lora":
        state_dict = torch.load(args.input_ckpt, weights_only=False)["state_dict"]
        state_dict = load_lora_weights(state_dict, args.lora_ckpt, args.model_key)
        torch.save(state_dict, args.output_path)
    elif args.command == "load_finetuned":
        state_dict = torch.load(args.input_ckpt, weights_only=False)["state_dict"]
        state_dict = load_finetuned_weights(
            state_dict, args.finetuned_ckpt, args.model_key
        )
        torch.save(state_dict, args.output_path)
    elif args.command == "merge_lora":
        model = torch.load(args.model_ckpt, weights_only=False)
        merge_lora(model)
        torch.save(model.state_dict(), args.output_path)


if __name__ == "__main__":
    main()
