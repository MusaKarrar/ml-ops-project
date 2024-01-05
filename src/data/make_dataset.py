if __name__ == "__main__":
    # Get the data and process it
    import torch
    from torchvision import transforms

    train_images = torch.empty((0, 784), dtype=torch.float32).reshape(0, 28, 28)  # 28x28 = 784 is num of features.
    train_targets = torch.empty((0), dtype=torch.long)
    for i in range(5):
        train_images = torch.cat((train_images, torch.load(f"data/raw/corruptmnist/train_images_{i}.pt")), 0)
        train_targets = torch.cat((train_targets, torch.load(f"data/raw/corruptmnist/train_target_{i}.pt")), 0)

    test_images = torch.load(f"data/raw/corruptmnist/test_images.pt")
    test_targets = torch.load(f"data/raw/corruptmnist/test_target.pt")
    test_images = test_images.type(torch.float32)
    test_targets = test_targets.type(torch.long)

    transform_norm = transforms.Compose([transforms.Normalize(0, 1)])
    train_images = transform_norm(train_images).flatten(1, 2)
    test_images = transform_norm(test_images).flatten(1, 2)

    torch.save(train_images, "data/processed/train_images.pt")
    torch.save(test_images, "data/processed/test_images.pt")
    torch.save(train_targets, "data/processed/train_targets.pt")
    torch.save(test_targets, "data/processed/test_targets.pt")

    pass
