import torch
from torch.utils.data import DataLoader
from SoundClasses.SoundDataset import SoundDataset
import os
from hyperparams import hyperparams as hp


def dataloader_exists(path):
    return os.path.isfile(path)

def get_model(dataloader):
    if hp.model_name == "CNN":
        from models.CNN import Trainer, CNN
        model = CNN()
        trainer = Trainer(dataloader, model)
    elif hp.model_name == "RNN":
        from models.RNN import Trainer, RNN
        model = RNN(hp.n_features, 64, 3, hp.num_classes, hp.device, classes=None)
        trainer = Trainer(dataloader, model)
    elif hp.model_name == "AudioEncoder":
        from models.AudioEncoder import Trainer, AudioEncoder
        model = AudioEncoder()
        trainer = Trainer(dataloader, model)
    elif hp.model_name == "GAN":
        from models.GAN import Gan, Trainer
        model = Gan()
        trainer = Trainer(dataloader, model)
    elif hp.model_name == "Test":
        from models.TestModul import TestModule, Trainer
        model = TestModule()
        trainer = Trainer(dataloader, model)
    return model, trainer

def new_dataloader():
    dataset = SoundDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=hp.batch_size, shuffle=True, num_workers=0)
    torch.save(dataloader, hp.dataloader_path)
    return dataloader

def get_dataloader():
    if dataloader_exists(hp.dataloader_path):
        dataloader = torch.load(hp.dataloader_path)
        if dataloader.batch_size == hp.batch_size:
            print("dataloader found at: " + hp.dataloader_path)
        else:
            dataloader = new_dataloader()
    else:
        dataloader = new_dataloader()
    return dataloader

if __name__ == "__main__":
    print("prepare dataset...")
    dataloader = get_dataloader()

    model, trainer = get_model(dataloader)
    print("start training...")

    trainer.init_training()

    for epoch in range(hp.n_epochs):
        trainer.init_epoch()

        for data, label in trainer.dataloader:
            trainer.step(data, label)
            trainer.log_entry(label)

        if hp.verbose:
            trainer.verbose(epoch)

    torch.save(trainer.model.state_dict(), hp.model_path)