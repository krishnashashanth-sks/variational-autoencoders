import torch

def train(num_epochs,vqvae_model,device,optimizer,train_dataloader,val_dataloader):
    for epoch in range(num_epochs):
        vqvae_model.train() # Set model to training mode
        total_train_loss = 0
        total_train_perplexity = 0

        for batch_idx, (images, _) in enumerate(train_dataloader):
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            reconstructed_images, loss, perplexity, _ = vqvae_model(images)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_perplexity += perplexity.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_perplexity = total_train_perplexity / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Perplexity: {avg_train_perplexity:.2f}')

        # 4. Define the evaluation loop
        vqvae_model.eval() # Set model to evaluation mode
        total_val_loss = 0
        total_val_perplexity = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for batch_idx, (images, _) in enumerate(val_dataloader):
                images = images.to(device)

                reconstructed_images, loss, perplexity, _ = vqvae_model(images)

                total_val_loss += loss.item()
                total_val_perplexity += perplexity.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_perplexity = total_val_perplexity / len(val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {avg_val_perplexity:.2f}')

    print("--- VQ-VAE Training and Evaluation Complete ---")
