import torch
from get_model.model.model import get_finetune_motif

class InferenceModel:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the inference model with the provided checkpoint.
        """
        self.device = device
        self.model = self.load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def rename_keys(self, state_dict):
        """
        Rename the keys in the state dictionary.
        """
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key
            # Adjust keys according to the new model architecture
            new_key = new_key.replace("blocks.", "encoder.blocks.")
            new_key = new_key.replace("fc_norm.", "encoder.norm.")
            new_key = new_key.replace("head.", "head_exp.head.")
            new_key = new_key.replace("region_embed.proj.", "region_embed.embed.")
            
            new_state_dict[new_key] = state_dict[key]

        # Adjust the weight dimensions if needed
        # Uncomment the next line if the weight dimension needs to be changed
        # new_state_dict['region_embed.embed.weight'] = new_state_dict['region_embed.embed.weight'].unsqueeze(2)
        return new_state_dict

    def load_model(self, checkpoint_path):
        """
        Load the model from a checkpoint file.
        """
        # Instantiate the model using the factory function
        model = get_finetune_motif()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # If the checkpoint is a dictionary with a 'model' key, extract the state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Rename state dict keys to match the model's keys
        state_dict = self.rename_keys(state_dict)
        
        # Load the state dict into the model
        model.load_state_dict(state_dict, strict=False)
        
        return model

    def predict(self, *input_tensors):
        """
        Perform inference on the given input tensors.
        """
        with torch.no_grad():
            # Ensure that input tensors are on the correct device
            input_tensors = [tensor.to(self.device) for tensor in input_tensors]
            # Forward pass through the model
            outputs = self.model(*input_tensors)
            
            # Process outputs if necessary (e.g., apply softmax)
            # For now, we will just return the raw outputs
            return outputs

# Example usage
if __name__ == "__main__":
    checkpoint_path = '/manitou/pmg/projects/resources/get_interpret/pretrain_finetune_natac_fetal_adult.pth'
    inf_model = InferenceModel(checkpoint_path, 'cpu')

    # Create dummy data with correct shape
    # Replace these with real inputs
    peak = torch.randn(1, 200, 283).to(inf_model.device)
    seq = torch.randn(1, 200, 283, 4).to(inf_model.device)  # Dummy seq data
    tss_mask = torch.ones(1, 200).to(inf_model.device)  # Dummy TSS mask
    ctcf_pos = torch.ones(1, 200).to(inf_model.device)  # Dummy CTCF positions

    # Perform inference
    atac, exp = inf_model.predict(peak, seq, tss_mask, ctcf_pos)
    print("ATAC predictions:", atac)
    print("Expression predictions:", exp)
