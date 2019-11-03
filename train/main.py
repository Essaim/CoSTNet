from utils.data_contrainer import get_pretrain_dataloader, get_predict_dataloader
from utils.load_config import get_config
from model.CNN_AE import Model_Pretrain_CNN
from model.LSTM_Prediction import Model_Predict_LSTM
from train.train import train_pretrain, train_predict


def get_model(act, encoder=None, decoder=None, input_size=None, output_size=None):
    if act == 'pretrian':
        return Model_Pretrain_CNN(get_config("filters"))
    elif act == 'train':
        return Model_Predict_LSTM(encoder, decoder, input_size, output_size)
    else:
        print("act error")


def main():
    pretrain_datapath = f"data/{get_config('pretrain_data_path')}"
    predict_datapath = f"data/{get_config('predict_data_path')}"

    loss_func, opt = [], []
    pretrain_dataloader = get_pretrain_dataloader(pretrain_datapath, get_config("batch_size"), channel=0)

    pretrain_model = train_pretrain(model=get_model("pretrain"),
                                    data_loader=pretrain_dataloader,
                                    loss_func=loss_func,
                                    optimizer=opt,
                                    num_epochs=get_config("pretrain_epochs"),
                                    tensorboard_folder=get_config("xxx"),
                                    model_folder=get_config("xxx")
                                    )
    encoder, decoder = pretrain_model.encoder, pretrain_model.decoder

    predict_dataloader = get_predict_dataloader(xxxxxxx)
    predict_model = get_model("train",
                              encoder=encoder,
                              decoder=decoder,
                              input_size=get_config("width") * get_config("height"),
                              output_size=get_config("width") * get_config("height"))
    train_predict(model=predict_model,
                  data_loader=predict_dataloader,
                  loss_func=loss_func,
                  optimizer=opt,
                  num_epochs=get_config("train_epochs"),
                  tensorboard_folder=get_config("xxx"),
                  model_folder=get_config("xxx")
                  )


if __name__ == '__main__':
    main()
