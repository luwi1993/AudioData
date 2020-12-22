import torch
from torch.utils.data import DataLoader
from SoundClasses.SoundDataset import SoundDataset
import os
from hyperparams import hyperparams as hp


def dataloader_exists(path):
    return os.path.isfile(path)

def get_model(hp, dataloader):
    if hp.model_name == "CNN":
        from models.CNN import Trainer, CNN
        model = CNN(hp)
        trainer = Trainer(hp, dataloader, model)
    elif hp.model_name == "RNN":
        from models.RNN import Trainer, RNN
        model = RNN(hp.n_features, 64, 3, hp.num_classes, hp.device, classes=None)
        trainer = Trainer(hp, dataloader, model)
    elif hp.model_name == "AudioEncoder":
        from models.AudioEncoder import Trainer, AudioEncoder
        model = AudioEncoder()
        trainer = Trainer(hp, dataloader, model)
    return model, trainer

if __name__ == "__main__":

    print("prepare dataset...")
    if dataloader_exists(hp.dataloader_path):
        dataloader = torch.load(hp.dataloader_path)
        print("dataloader found at: "+ hp.dataloader_path)
    else:
        dataset = SoundDataset()
        dataloader = DataLoader(dataset=dataset, batch_size=hp.batch_size, shuffle=True, num_workers=0)
        torch.save(dataloader, hp.dataloader_path)

    model, trainer = get_model(hp, dataloader)
    print("start training...")
    trainer.train()
