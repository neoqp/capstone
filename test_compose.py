import os
import math
import argparse
import logging
from PIL import Image
import torch
import numpy as np
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import options.options as option
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy
import sys

def cal_psnr(sr_img, gt_img):
    sr_img = sr_img / 255.
    gt_img = gt_img / 255.
    return util.calculate_psnr(sr_img * 255, gt_img * 255)

def bit_error_rate_tensor(m1, m2):
    b1 = m1.flatten().to(torch.int)
    b2 = m2.flatten().to(torch.int)
    return torch.sum(b1 != b2).item() / len(b1)

def split_image_tensor(tensor, grid_size):
    C, H, W = tensor.shape
    h_step = H // grid_size
    w_step = W // grid_size
    return [
        tensor[:, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
        for i in range(grid_size)
        for j in range(grid_size)
    ]

def compose_image_tensor(patches, grid_size):
    rows = []
    for i in range(grid_size):
        row = torch.cat(patches[i*grid_size:(i+1)*grid_size], dim=2)
        rows.append(row)
    return torch.cat(rows, dim=1)

def extract_message_from_modified_lr(model, modified_lr_tensor):
    model.netG.eval()
    with torch.no_grad():
        # (1, C, H, W) 형태로 변경 + device 이동
        x = modified_lr_tensor.unsqueeze(0).to(model.device)

        # Quantization 적용 (원래 LR도 이 과정을 거침)
        x = model.Quantization(x)

        # message 추출
        out_x, out_x_h, out_z, recmessage = model.netG(x=x, rev=True)  # bit 모드에선 단일 값 반환

        # clamp + 이진화
        recmessage = torch.clamp(recmessage, -0.5, 0.5)
        recmessage[recmessage > 0] = 1
        recmessage[recmessage <= 0] = 0

        return recmessage.squeeze(0).to(torch.int)
    
def bit_difference(a_list, b_list):
    assert len(a_list) == len(b_list), "두 리스트의 길이가 같아야 합니다."
    
    a = torch.tensor(a_list, dtype=torch.int)
    b = torch.tensor(b_list, dtype=torch.int)
    
    diff = (a != b).int()  # 다르면 True -> 1, 같으면 False -> 0
    return diff.tolist()

def expand_bits_torch(bit_array):
    n = len(bit_array)
    size = int(n ** 0.5)
    assert size * size == n, "입력 배열의 길이는 제곱수여야 합니다."
    
    # 1차원 -> 2차원 tensor 변환
    grid = torch.tensor(bit_array, dtype=torch.int).reshape(size, size)
    
    # padding 추가 (가장자리 처리 간편하게 하기 위함)
    padded_grid = torch.nn.functional.pad(grid, (1, 1, 1, 1), mode='constant', value=0)
    
    # 확장된 결과 담을 tensor
    expanded_grid = padded_grid.clone()
    
    # 8방향 이동을 위한 커널 정의
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]
    
    for dx, dy in shifts:
        shifted = torch.roll(padded_grid, shifts=(dx, dy), dims=(0, 1))
        expanded_grid = torch.maximum(expanded_grid, shifted)
    
    # 패딩 제거하고 다시 flatten
    expanded_grid = expanded_grid[1:-1, 1:-1]
    return expanded_grid.flatten().tolist()

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--ckpt', type=str, default='checkpoints/clean.pth')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size for splitting image (e.g., 2 for 2x2)')
    parser.add_argument('--resize', type=int, default=512)
    args = parser.parse_args()
    sys.stdout = Logger("results/exp.txt")
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    val_set = create_dataset(opt['datasets']['val'])
    val_loader = create_dataloader(val_set, opt['datasets']['val'], opt, None)
    model = create_model(opt)
    model.load_test(args.ckpt)
    transform = transforms.Resize((args.resize, args.resize))
    for image_id, val_data in enumerate(val_loader):
        
        lq_tensor = val_data.get('LQ').squeeze()
        gt_tensor = val_data.get('GT').squeeze()
        # composed_img = util.tensor2img(lq_tensor)
        # util.save_img(composed_img, f'results/lq_{image_id}.png')

        lq_tensor = transform(lq_tensor)
        gt_tensor = transform(gt_tensor)

        lq_quads = split_image_tensor(lq_tensor, args.grid_size)
        gt_quads = split_image_tensor(gt_tensor, args.grid_size)

        embedded_quads = []
        original_messages = []
        recovered_messages = []

        for i, quad in enumerate(lq_quads):

            lq = quad.unsqueeze(0).unsqueeze(0).unsqueeze(0)        # (1,1,1,C,H,W)
            gt = gt_quads[i].unsqueeze(0).unsqueeze(0)              # (1,1,C,H,W)

            data = {
                'LQ': lq, # <- 이새끼 쓸모없음
                'GT': gt
            }

            model.feed_data(data)
            model.test(image_id * 1000 + i)
            visuals = model.get_current_visuals()
            # LR = visuals.get('LR')[0].squeeze().unsqueeze(0) # 워터마크 들어간 애

            # data = {
            #     'LQ': LR,
            #     'GT': gt
            # }

            # model.feed_data(data)
            # rec_msg = extract_message_from_modified_lr(model, LR.squeeze()) # (64) tensor
            rec_msg = visuals['recmessage'][0]
            orig_msg = visuals['message'][0]    # (1, 64) tensor
                

            orig_str = ''.join(str(int(b)) for b in orig_msg.flatten().tolist())
            rec_str = ''.join(str(int(b)) for b in rec_msg.flatten().tolist())

            # recovered_messages.append(rec_msg)
            # original_messages.append(orig_msg)
            embedded_quads.append(visuals['LR'].squeeze())


            orig_str = ''.join(str(int(b)) for b in orig_msg.flatten().tolist())
            rec_str = ''.join(str(int(b)) for b in rec_msg.flatten().tolist())
            ber = bit_error_rate_tensor(rec_msg, orig_msg)

            print(f"Message (Quad {i}):     {orig_str}")
            print(f"Recovered (Quad {i}):   {rec_str}")
            print(f"Image {image_id} Quad {i}: BER = {ber*100:.4f}%")

        composed = compose_image_tensor(embedded_quads, args.grid_size)
        composed_img = util.tensor2img(composed)
        # util.save_img(composed_img, f'results/method1_{args.resize}/compose/{image_id}.png')

if __name__ == '__main__':
    main()
