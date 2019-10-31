



from utils.data_contrainer import get_pretrain_dataloader, get_predict_dataloader
from utils.load_config import get_config


def get_model():
    return




def main():
    pretrain_datapath = f"data/{get_config('pretrain_data_path')}"
    predict_datapath = f"data/{get_config('predict_data_path')}"


    pretrain_dataloader = get_pretrain_dataloader(pretrain_datapath, get_config("batch_size"), channel = 0)



    predict_dataloader = get_predict_dataloader()







if __name__ == '__main__':
    main()