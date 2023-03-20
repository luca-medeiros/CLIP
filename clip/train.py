
import lightning as L
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
import torchvision
from transformers import BertTokenizer

from clip.data import food101
from clip.models.clip_model import CLIPModel


def main():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    vocab_size = tokenizer.vocab_size
    # create model
    model = CLIPModel(vocab_size)

    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    # setup data
    train_loader, test_loader = food101()

    # setup Fabric
    fabric = Fabric(accelerator="cuda", devices=4, strategy="ddp")
    fabric.launch()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    # train the model
    for epoch in range(2):
        fabric.print("Epoch:", epoch)
        for i, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            images, class_ids = batch
            texts = [train_loader.dataset.classes[idx] for idx in class_ids]
            tokens = tokenizer.batch_encode_plus(texts, add_special_tokens=True, padding='max_length', max_length=30, return_tensors='pt')
            optimizer.zero_grad()

            # forward + loss
            logits_per_image = model(images, tokens['input_ids'])
            loss = F.cross_entropy(
                logits_per_image,
                torch.arange(len(logits_per_image), device=model.device, dtype=torch.long)) +\
                F.cross_entropy(
                    logits_per_image.t(),
                    torch.arange(len(logits_per_image), device=model.device, dtype=torch.long))
            loss = loss / 2

            # backward + optimize
            fabric.backward(loss)
            optimizer.step()

            if i % 100 == 0:
                fabric.print("train_loss", float(loss))
                fabric.log("train_loss", loss)

if __name__ == "__main__":
    main()
