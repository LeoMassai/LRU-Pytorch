import torch
import torch.nn as nn
import math


class LRU(nn.Module):
    def __init__(self, in_features, out_features, state_features, rmin=0.9, rmax=1, max_phase=6.283):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))

        # Initializing complex parameters B and C
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))

        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))

        # Initial state is set to complex zeros
        self.state = torch.complex(torch.zeros(state_features), torch.zeros(state_features))

    def forward(self, input):
        # Ensure device consistency
        self.state = self.state.to(self.B.device)

        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im).to(self.state.device)

        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.state.device)

        # Use torch.Size to ensure static inference of the output shape
        output_shape = torch.Size([input.size(0), input.size(1), self.out_features])
        output = torch.empty(output_shape, device=self.B.device)

        if input.dim() == 3:  # Input shape: (Batches, Seq_length, Input size)
            for i in range(input.size(0)):
                for j in range(input.size(1)):
                    step = input[i, j]

                    # Compute the new state
                    self.state = Lambda * self.state + gammas * (self.B @ step.to(dtype=self.B.dtype))

                    # Check dimensions before performing operations
                    print(
                        f"self.state shape: {self.state.shape}, (self.C @ self.state).real shape: {(self.C @ self.state).real.shape}, step shape: {step.shape}, D @ step shape: {self.D @ step.to(dtype=self.D.dtype).shape}")

                    # Ensure the operation is valid for complex addition
                    out_step = (self.C @ self.state).real + (
                                self.D @ step.to(dtype=self.D.dtype))  # Ensure step is real for D
                    output[i, j] = out_step

                # Reset state after processing each batch
                self.state = torch.complex(torch.zeros_like(self.state.real), torch.zeros_like(self.state.real))

        elif input.dim() == 2:  # Input shape: (Seq_length, Input size)
            for i in range(input.size(0)):
                step = input[i]
                self.state = Lambda * self.state + gammas * (self.B @ step.to(dtype=self.B.dtype))
                out_step = (self.C @ self.state).real + (
                            self.D @ step.to(dtype=self.D.dtype))  # Ensure step is real for D
                output[i] = out_step

            # Reset state after processing the sequence
            self.state = torch.complex(torch.zeros_like(self.state.real), torch.zeros_like(self.state.real))

        return output


# Example Usage
if __name__ == "__main__":
    # Define dummy input: 3D tensor (Batch, Seq_length, Input size)
    batch_size = 2
    seq_length = 10
    input_size = 4  # Example input size, adjust as necessary
    u = torch.randn(batch_size, seq_length, input_size, dtype=torch.complex64, requires_grad=True)

    # Initialize the model
    RNN = LRU(in_features=input_size, out_features=5, state_features=3)

    # Script the model using torch.jit.script
    scripted_RNN = torch.jit.script(RNN)

    # Forward pass using the scripted model
    output = scripted_RNN(u)

    # Define a loss function, ensuring it outputs a real scalar value
    loss = output.real.sum()  # Use .real to ensure loss is real-valued

    # Backward pass with exception handling for potential type mismatches
    try:
        loss.backward()
    except RuntimeError as e:
        print(f"Error during backward pass with scripted module: {e}")
        # Ensure gradient types match; convert if necessary
        if u.grad is not None and u.grad.dtype != u.dtype:
            u.grad = u.grad.to(dtype=u.dtype)

    # Check if gradients are properly set
    print("Gradients:", u.grad)

