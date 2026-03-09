import torch
from tqdm import tqdm


def train(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    bar = tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)
    total_loss = 0.0

    for i, batch in enumerate(bar):
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(batch)

        # CLIPLoss(bert_embs, gnn_embs, labels)
        loss = criterion(*output, batch.label)

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # ✅ OneCycleLR 必须每个 step 调一次（你的 config 当前不是 onecycle，但这样更健壮）
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        bar.set_postfix(
            l=f"{total_loss / (i + 1):.3f}",
            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
        )

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    bar = tqdm(dataloader, desc="Val", leave=False, dynamic_ncols=True)
    total_loss = 0.0

    for i, batch in enumerate(bar):
        batch = batch.to(device)

        with torch.inference_mode():
            output = model(batch)
            loss = criterion(*output, batch.label)
            total_loss += loss.item()

        bar.set_postfix(l=f"{total_loss / (i + 1):.3f}")

    return total_loss / len(dataloader)
