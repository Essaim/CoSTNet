import os
import shutil
from torch import optim
import torch

from utils.data_contrainer import get_spatio_dataloader, get_temporal_dataloader
from utils.load_config import get_config
from model.CNN_AE import Model_Spatio_CNN
from model.LSTM_Prediction import Model_Temporal_LSTM
from train.train_main import train_spatio, train_temporal, train_model
from utils.loss import RMSELoss
from utils.normalization import preprocess_normalize


def get_model(model_type, encoder=None, decoder=None, input_size=None, output_size=None):
    if model_type == 'spatio':
        return Model_Spatio_CNN(get_config("filters"))
    elif model_type == 'temporal':
        return Model_Temporal_LSTM(encoder, decoder, input_size, output_size)
    else:
        raise ValueError("Unknown model type.")


def create_loss(loss_type):
    if loss_type == 'rmse_loss':
        return RMSELoss()
    else:
        raise ValueError("Unknown loss type.")


def create_opt(par, lr, opt_type):
    if opt_type == "Adam":
        return optim.Adam(par, lr)
    elif opt_type == "SGD":
        return optim.SGD(par, lr, momentum=0.9)
    else:
        raise NotImplementedError()


def main():
    model_folder = f"../save/{get_config('model_name')}"
    tensorboard_folder = f"../run/{get_config('model_name')}"

    shutil.rmtree(model_folder, ignore_errors=True)
    os.makedirs(model_folder, exist_ok=True)
    shutil.rmtree(tensorboard_folder, ignore_errors=True)
    os.makedirs(tensorboard_folder, exist_ok=True)

    spatio_data_path = f"../data/{get_config('spatio_data_path')}"
    temporal_data_path = f"../data/{get_config('temporal_data_path')}"

    spaio_data_normal, temporal_data_normal, normal_st = preprocess_normalize(spatio_data_path, temporal_data_path, 0)

    loss_func = create_loss(get_config("loss_type"))
    spatio_dataloader = get_spatio_dataloader(datapath=spatio_data_path, normal=normal_st,
                                              batch_size=get_config("spatio_batch_size"), channel=0,
                                              pre_train_len=get_config("spatio_train_len"))
    spatio_model = get_model("spatio").to(get_config("device"))
    spatio_opt = create_opt(spatio_model.parameters(), get_config("learning_rate"), get_config("opt_type"))

    spatio_model = train_model(model=spatio_model,
                               data_loader=spatio_dataloader,
                               phases=['train', 'validate'],
                               normal=normal_st,
                               loss_func=loss_func,
                               optimizer=spatio_opt,
                               num_epochs=get_config("spatio_epochs"),
                               tensorboard_folder=tensorboard_folder,
                               model_folder_name=f"{model_folder}/ae_model.pkl")


    # spatio_model.load_state_dict(torch.load(f"{model_folder}/ae_model.pkl")['model_state_dict'])

    encoder, decoder = spatio_model.encoder, spatio_model.decoder

    temporal_dataloader = get_temporal_dataloader(datapath=temporal_data_path,
                                                  normal=normal_st,
                                                  batch_size=get_config("temporal_batch_size"),
                                                  encoder=encoder,
                                                  channel=0,
                                                  depend_list=get_config("temporal_depend_list"))
    temporal_model = get_model("temporal",
                               encoder,
                               decoder,
                               get_config("lstm_width"),
                               get_config("lstm_height")).to(get_config("device"))
    temporal_opt = create_opt(temporal_model.parameters(), get_config("learning_rate"), get_config("opt_type"))
    # temporal_opt= optim.Adam(temporal_model.parameters(), get_config("learning_rate"), weight_decay=get_config("weight_decay"))
    train_model(model=temporal_model,
                data_loader=temporal_dataloader,
                phases=['train', 'validate', 'test'],
                normal=normal_st,
                loss_func=loss_func,
                optimizer=temporal_opt,
                num_epochs=get_config("temporal_epochs"),
                tensorboard_folder=tensorboard_folder,
                model_folder_name=f"{model_folder}/best_model.pkl")


if __name__ == '__main__':
    main()
