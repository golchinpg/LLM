import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Source.utils import calculate_loss_loader, generate_and_print_samples, calculate_loss_batch

def train_model(model, train_loader, val_loader, optimizer, scheduler,  epochs,
                eval_freq, eval_iter, start_context, tokenizer, device):
    
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    max_grad_norm = 1.0
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calculate_loss_batch(model, input_batch, target_batch, device)
            loss.backward() # Calculate loss gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Clip gradients to avoid exploding gradients
            optimizer.step() # Update model weights using loss gradients
            scheduler.step() # Update learning rate
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_samples(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calculate_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss
