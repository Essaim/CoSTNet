



from utils.data_contrainer import get_pretrain_dataloader, get_predict_dataloader
from utils.load_config import get_config





def main():
    pretrain_datapath = f"data/{get_config('pretrain_datapath')}"
    predict_datapath = f"data/{get_config('predict_datapath')}"


    pretrain_dataloader = get_predict_dataloader(pretrain_datapath, get_config("batch_size"), channel = 0)


    predict_dataloader = get_predict_dataloader()







if __name__ == '__main__':
    main()