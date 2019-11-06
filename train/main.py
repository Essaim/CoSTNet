import os
import shutil
from torch import optim
import torch

from utils.data_contrainer import get_spatio_dataloader, get_temporal_dataloader
from utils.load_config import get_config
from model.CNN_AE import Model_Spatio_CNN
from model.LSTM_Prediction import Model_Temporal_LSTM
from train.train_main import train_model
from utils.loss import RMSELoss
from utils.normalization import preprocess_normalize


def get_model(model_type, encoder=None, decoder=None, input_size=None, output_size=None, channel_num=None):
    if model_type == 'spatio':
        return Model_Spatio_CNN(get_config("filters"))
    elif model_type == 'temporal':
        return Model_Temporal_LSTM(encoder, decoder, input_size, output_size, channel_num)
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


def get_spatio_model(spatio_data_path, temporal_data_path, channel, tensorboard_folder, model_folder_name):

    spaio_data_normal, temporal_data_normal, normal_st = preprocess_normalize(spatio_data_path, temporal_data_path,
                                                                              channel)
    loss_func = create_loss(get_config("loss_type"))
    spatio_dataloader = get_spatio_dataloader(datapath=spatio_data_path,
                                              normal=normal_st,
                                              batch_size=get_config("spatio_batch_size"),
                                              channel=channel,
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
                               model_folder_name=model_folder_name)
    # spatio_model.load_state_dict(torch.load(f"{model_folder_name}")['model_state_dict'])

    return {'model': spatio_model, 'normal': normal_st}


def get_ednd(bike_pick, bike_drop, taxi_pick, taxi_drop, bike_temporal_data_path, taxi_temporal_data_path):
    container = [bike_pick, bike_drop, taxi_pick, taxi_drop]

    encoder = [each['model'].encoder for each in container]
    decoder = [each['model'].decoder for each in container]
    normal_st = [each['normal'] for each in container]
    temporal_data_path = [bike_temporal_data_path, bike_temporal_data_path, taxi_temporal_data_path,
                          taxi_temporal_data_path]

    return encoder, decoder, normal_st, temporal_data_path













def main():
    model_folder = f"../save/{get_config('model_name')}"
    tensorboard_folder = f"../run/{get_config('model_name')}"

    shutil.rmtree(model_folder, ignore_errors=True)
    os.makedirs(model_folder, exist_ok=True)
    shutil.rmtree(tensorboard_folder, ignore_errors=True)
    os.makedirs(tensorboard_folder, exist_ok=True)

    bike_spatio_data_path = f"../data/{get_config('bike_spatio_data_name')}"
    bike_temporal_data_path = f"../data/{get_config('bike_temporal_data_name')}"
    taxi_spatio_data_path = f"../data/{get_config('taxi_spatio_data_name')}"
    taxi_temporal_data_path = f"../data/{get_config('taxi_temporal_data_name')}"

    bike_pick = get_spatio_model(spatio_data_path=bike_spatio_data_path,
                                 temporal_data_path=bike_temporal_data_path,
                                 channel=0,
                                 tensorboard_folder=tensorboard_folder,
                                 model_folder_name=f"{model_folder}/bike_pick_ae_model.pkl")
    bike_drop = get_spatio_model(spatio_data_path=bike_spatio_data_path,
                                 temporal_data_path=bike_temporal_data_path,
                                 channel=1,
                                 tensorboard_folder=tensorboard_folder,
                                 model_folder_name=f"{model_folder}/bike_drop_ae_model.pkl")
    taxi_pick = get_spatio_model(spatio_data_path=taxi_spatio_data_path,
                                 temporal_data_path=taxi_temporal_data_path,
                                 channel=0,
                                 tensorboard_folder=tensorboard_folder,
                                 model_folder_name=f"{model_folder}/taxi_pick_ae_model.pkl")
    taxi_drop = get_spatio_model(spatio_data_path=taxi_spatio_data_path,
                                 temporal_data_path=taxi_temporal_data_path,
                                 channel=1,
                                 tensorboard_folder=tensorboard_folder,
                                 model_folder_name=f"{model_folder}/taxi_drop_ae_model.pkl")






    encoder, decoder, normal_st, temporal_data_path = get_ednd(bike_pick=bike_pick,
                                                               bike_drop=bike_drop,
                                                               taxi_pick=taxi_pick,
                                                               taxi_drop=taxi_drop,
                                                               bike_temporal_data_path=bike_temporal_data_path,
                                                               taxi_temporal_data_path=taxi_temporal_data_path)

    loss_func = create_loss(get_config("loss_type"))

    temporal_dataloader = get_temporal_dataloader(datapath=temporal_data_path,
                                                  normal=normal_st,
                                                  batch_size=get_config("temporal_batch_size"),
                                                  encoder=encoder,
                                                  channel=get_config("channel_num"),
                                                  depend_list=get_config("temporal_depend_list"))
    temporal_model = get_model("temporal",
                               encoder,
                               decoder[get_config("channel_num")],
                               get_config("lstm_width"),
                               get_config("lstm_height"),
                               get_config("channel_total")).to(get_config("device"))
    temporal_opt = create_opt(temporal_model.parameters(), get_config("learning_rate"), get_config("opt_type"))
    train_model(model=temporal_model,
                data_loader=temporal_dataloader,
                phases=['train', 'validate', 'test'],
                normal=normal_st[get_config("channel_num")],
                loss_func=loss_func,
                optimizer=temporal_opt,
                num_epochs=get_config("temporal_epochs"),
                tensorboard_folder=tensorboard_folder,
                model_folder_name=f"{model_folder}/best_model.pkl")


if __name__ == '__main__':
    main()
