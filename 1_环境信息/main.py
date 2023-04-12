import torch
if __name__ == '__main__':
    # torch版本
    print(torch.__version__)
    # 是否支持cuda
    print(torch.cuda.is_available())