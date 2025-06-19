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
import torch.nn.functional as F
import torchvision.transforms as transforms
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

def insert_black_in_patch(lq_tensor, grid_size, i):
    C, H, W = lq_tensor.shape
    h_step = H // grid_size
    w_step = W // grid_size

    row = i // grid_size
    col = i % grid_size

    # 마스크 생성
    mask = torch.zeros_like(lq_tensor)
    h_start = row * h_step
    h_end = (row + 1) * h_step
    w_start = col * w_step
    w_end = (col + 1) * w_step

    mask[:, h_start:h_end, w_start:w_end] = 1

    return lq_tensor * mask

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
    sys.stdout = Logger("results/exp4.txt")

    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)
    grid_size = args.grid_size

    val_set = create_dataset(opt['datasets']['val'])
    val_loader = create_dataloader(val_set, opt['datasets']['val'], opt, None)
    model = create_model(opt)
    model.load_test(args.ckpt)
    transform = transforms.Resize((args.resize, args.resize))
    for image_id, val_data in enumerate(val_loader):
        lq_tensor = val_data.get('LQ').squeeze()
        gt_tensor = val_data.get('GT').squeeze()
        lq_tensor = transform(lq_tensor)
        gt_tensor = transform(gt_tensor)

        LQ_tensor = lq_tensor
        GT_tensor = gt_tensor
        composed_quadrants = []
        for i in range(grid_size * grid_size):
            LQ = insert_black_in_patch(LQ_tensor.squeeze(), grid_size, i).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
            GT = GT_tensor.unsqueeze(0).unsqueeze(0)
            data = {
                'LQ': LQ,
                'GT': GT
            }
            model.feed_data(data)
            model.test(image_id * 100 + i)
            visuals = model.get_current_visuals()
            orig_msg = visuals['message'][0]
            LR = visuals.get('LR')[0].squeeze()
            
            insert_black_LR_patch = insert_black_in_patch(LR, grid_size, i) # 원본사이즈 동일 / 워터마크 넣은 애 / 검은색 포함
            after_LR_patches = split_image_tensor(LR, grid_size) # 원본보다 작음 / 워터마크 넣고 자른 조각

            row = i // grid_size
            col = i % grid_size
            C, H, W = LR.shape
            h_step = H // grid_size
            w_step = W // grid_size

            LR_patch = after_LR_patches[row * grid_size + col]
            composed_quadrants.append(LR_patch)

            # full_img_np = util.tensor2img(insert_black_LR_patch)
            # save_path = f'results/first_insert_{image_id}_{row}_{col}.png'
            # util.save_img(full_img_np, save_path)

            data = {
                # 'LQ': insert_black_LR_patch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                'LQ': LR_patch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                'GT': GT
            }

            model.feed_data(data)
            rec_msg = extract_message_from_modified_lr(model,insert_black_LR_patch)
            # mask, rec_msg = model.image_recovery(0.5)

            orig_str = ''.join(str(int(b)) for b in orig_msg.flatten().tolist())
            rec_str = ''.join(str(int(b)) for b in rec_msg.flatten().tolist())
            ber = bit_error_rate_tensor(rec_msg, orig_msg)

            print(f"Message (Quad {i}):     {orig_str}")
            print(f"Recovered (Quad {i}):   {rec_str}")
            print(f"Image {image_id} Quad {i}: BER = {ber*100:.4f}%")
            

        composed_image = compose_image_tensor(composed_quadrants, grid_size)
        composed_img_np = util.tensor2img(composed_image)
        util.save_img(composed_img_np, f'results/method2_{args.resize}/compose/{image_id}.png')


if __name__ == '__main__':
    main()
