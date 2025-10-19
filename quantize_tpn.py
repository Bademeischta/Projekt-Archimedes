import torch
from torch.quantization import quantize_dynamic
from src.archimedes.model import TPN

def main():
    # Load the trained TPN model
    # In a real scenario, you would load your trained model weights here
    model = TPN()
    # model.load_state_dict(torch.load("tpn_weights.pth"))

    # Apply dynamic quantization
    quantized_model = quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(quantized_model.state_dict(), "quantized_tpn_weights.pth")
    print("TPN model quantized and saved to 'quantized_tpn_weights.pth'")

if __name__ == "__main__":
    main()
