import os
import torch
import csv
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
import hist
import numpy as np


real_im_dir = '/home/dan/Projects/2020/Data/Self/Aligned/'
fake_ours_dir = './no_attn/'
fake_BG_dir = '/home/dan/Projects/2020/BeautyGAN-master/'
fake_DMT_dir = '/home/dan/Projects/2020/DMT-master/'
mask_dir = '/home/dan/Projects/2020/Data/Self/Aligned/Masks/'

class Test():
    def __init__(self):
        self.criterion = torch.nn.L1Loss()

        self.real_im_dir = '/home/dan/Projects/2020/Data/Self/Aligned/'
        self.fake_ours_dir = './YT'
        self.fake_BG_dir = '/home/dan/Projects/2020/DMT-master'
        self.fake_DMT_dir = '/home/dan/Projects/2020/DMT-master'
        self.mask_dir = '/home/dan/Projects/2020/Data/Self/Aligned/Masks/'

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        save_image(mask_B_temp, 'masked_test.png')
        mask_A_temp[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_face[: ,: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
        mask_B_temp[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_face[: ,: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
        #mask_A_temp = hist.to_var(mask_A_temp, requires_grad=False)
        #mask_B_temp = hist.to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp


    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        #mask_A = to_var(mask_A, requires_grad=False)
        #mask_B = to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2

    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        #input_data = (hist.de_norm(input_data) * 255).squeeze()
        #target_data = (hist.de_norm(target_data) * 255).squeeze()
        save_image(mask_src, 'maskinput.png')
        save_image(mask_tar, 'masktarget.png')
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        dstImg = (input_masked.data).cpu().clone()
        refImg = (target_masked.data).cpu().clone()
        #input_match = hist.histogram_matching(input_masked, target_masked, index)
        input_masked = hist.to_var(input_masked, requires_grad=False)
        target_masked = hist.to_var(target_masked, requires_grad=False)
        save_image(input_data, 'input.png')
        save_image(target_data, 'target.png')
        save_image(input_masked, 'masked_input.png')
        save_image(target_masked, 'masked_target.png')
        #input_match = hist.to_var(input_match, requires_grad=False)
        loss = self.criterion(input_masked, target_masked)
        return loss

    def load_ims_masks(self, i):
        tr = TT.Resize(256)

        real_A_path = real_im_dir + str(i+1) + '_144.png'
        real_B_path = real_im_dir + str(i+1) + '_m_144.png'
        fake_ours_path = fake_ours_dir + str(i+1) + '_144.png'
        fake_BG_path = fake_BG_dir + str(i+1) + '.png'
        fake_DMT_path = fake_DMT_dir + str(i+1) + '.png'
        mask_A_path = mask_dir + str(i+1) + '_1080.png'
        mask_B_path = mask_dir + str(i+1) + '_m_1080.png'

        #load images
        real_A = Image.open(real_A_path)
        real_A = TF.to_tensor(real_A)
        real_A.unsqueeze(0)

        real_B = Image.open(real_B_path)
        real_B = TF.to_tensor(real_B)
        real_B.unsqueeze(0)

        fake_ours = Image.open(fake_ours_path)
        fake_ours = tr(fake_ours)
        fake_ours= TF.to_tensor(fake_ours)
        fake_ours.unsqueeze(0)

        fake_BG = Image.open(fake_BG_path)
        fake_BG = tr(fake_BG)
        fake_BG= TF.to_tensor(fake_BG)
        fake_BG.unsqueeze(0)
        
        fake_DMT = Image.open(fake_DMT_path)
        fake_DMT = tr(fake_DMT)
        fake_DMT= TF.to_tensor(fake_DMT)
        fake_DMT.unsqueeze(0)

        #load masks
        mask_A = Image.open(mask_A_path)
        mask_A = TF.to_tensor(mask_A)*255

        mask_B = Image.open(mask_B_path)
        mask_B = TF.to_tensor(mask_B)*255
        return real_A, real_B, fake_ours, fake_BG, fake_DMT, mask_A.round().unsqueeze(0), mask_B.round().unsqueeze(0)

        

    def run(self):
        our_eyes = []
        our_eye_r = []
        our_face = []
        our_lips = []
        our_total = []
        our_total2 = []
        
        
        BG_eyes = []
        BG_face = []
        BG_lips = []
        BG_total = []
        BG_total2 = []
        
        DMT_eyes = []
        DMT_face = []
        DMT_lips = []
        DMT_total = []
        DMT_total2 = []

        for i in range(10):
            real_A, real_B, fake_ours, fake_BG, fake_DMT, mask_A, mask_B = self.load_ims_masks(i)

            mask_A_lip = (mask_A == 11).float() + (mask_A == 12).float()
            mask_B_lip = (mask_B == 11).float() + (mask_B == 12).float()
            mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)

            mask_A_skin = (mask_A == 1).float() + (mask_A == 2).float()
            mask_B_skin = (mask_B == 1).float() + (mask_B == 2).float()
            mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)

            mask_A_eye_left = (mask_A == 4).float()
            mask_A_eye_right = (mask_A == 5).float()
            mask_B_eye_left = (mask_B == 4).float()
            mask_B_eye_right = (mask_B == 5).float()
            mask_A_face = (mask_A == 1).float() + (mask_A == 6).float()
            mask_B_face = (mask_B == 1).float() + (mask_B == 6).float()

            if (mask_A_eye_left > 0).any() and (mask_B_eye_left > 0).any() and \
                    (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any():
                mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
                mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)
            mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
                self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
            mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
                self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)


            our_loss_lip = self.criterionHis(real_B, fake_ours, mask_A_lip, mask_B_lip, index_B_lip)
            our_loss_skin = self.criterionHis(real_B, fake_ours, mask_A_skin, mask_B_skin, index_B_skin)
            our_loss_l_eye = self.criterionHis(real_B, fake_ours, mask_A_eye_left, mask_B_eye_left, index_B_eye_left)
            our_loss_r_eye = self.criterionHis(real_B, fake_ours, mask_A_eye_right, mask_B_eye_right, index_B_eye_right)

            BG_loss_lip = self.criterionHis(real_B, fake_BG, mask_A_lip, mask_B_lip, index_B_lip)
            BG_loss_skin = self.criterionHis(real_B, fake_BG, mask_A_skin, mask_B_skin, index_B_skin)
            BG_loss_l_eye = self.criterionHis(real_B, fake_BG, mask_A_eye_left, mask_B_eye_left, index_B_eye_left)
            BG_loss_r_eye = self.criterionHis(real_B, fake_BG, mask_A_eye_right, mask_B_eye_right, index_B_eye_right)

            DMT_loss_lip = self.criterionHis(real_B, fake_DMT, mask_A_lip, mask_B_lip, index_B_lip)
            DMT_loss_skin = self.criterionHis(real_B, fake_DMT, mask_A_skin, mask_B_skin, index_B_skin)
            DMT_loss_l_eye = self.criterionHis(real_B, fake_DMT, mask_A_eye_left, mask_B_eye_left, index_B_eye_left)
            DMT_loss_r_eye = self.criterionHis(real_B, fake_DMT, mask_A_eye_right, mask_B_eye_right, index_B_eye_right)

            lip_pixels = mask_A_lip.nonzero().size()[0]
            skin_pixels = mask_A_skin.nonzero().size()[0]
            l_eye_pixels = mask_A_eye_left.nonzero().size()[0]
            r_eye_pixels = mask_A_eye_right.nonzero().size()[0]
            eye_pixels = l_eye_pixels + r_eye_pixels
            head_pixels = lip_pixels + skin_pixels + l_eye_pixels + r_eye_pixels
            
            
            #print(i)
            #print(our_loss_lip.item(), BG_loss_lip.item(), DMT_loss_lip.item())
            #print(our_loss_skin.item(), BG_loss_skin.item(), DMT_loss_skin.item())
            #print(our_loss_l_eye.item()+our_loss_r_eye.item(), BG_loss_l_eye.item()+BG_loss_r_eye.item(), DMT_loss_l_eye.item()+DMT_loss_r_eye.item())


            our_tot = head_pixels/lip_pixels * our_loss_lip + head_pixels/skin_pixels * our_loss_skin + head_pixels/eye_pixels*(our_loss_l_eye + our_loss_r_eye)
            DMT_tot = head_pixels/lip_pixels * DMT_loss_lip + head_pixels/skin_pixels * DMT_loss_skin + head_pixels/eye_pixels*(DMT_loss_l_eye + DMT_loss_r_eye)
            BG_tot = head_pixels/lip_pixels * BG_loss_lip + head_pixels/skin_pixels * BG_loss_skin + head_pixels/eye_pixels*(BG_loss_l_eye + BG_loss_r_eye)
            print(our_tot.item(), BG_tot.item(), DMT_tot.item())

            #our_tot2 =  our_loss_lip + our_loss_skin  + head_pixels/r_eye_pixels*our_loss_r_eye
            #DMT_tot2 =  DMT_loss_lip + DMT_loss_skin  + head_pixels/r_eye_pixels*DMT_loss_r_eye
            #BG_tot2 = BG_loss_lip + BG_loss_skin + BG_loss_l_eye + BG_loss_r_eye
            #print(our_tot2.item(), BG_tot2.item(), DMT_tot2.item())



            our_eyes.append(head_pixels/eye_pixels * (our_loss_l_eye.item() + our_loss_r_eye.item()))
            our_face.append(head_pixels/skin_pixels * our_loss_skin.item())
            our_lips.append(head_pixels/lip_pixels * our_loss_lip.item())
            our_total.append(our_tot.item())
            #our_total2.append(our_tot2.item())
            DMT_eyes.append(head_pixels/eye_pixels * (DMT_loss_l_eye.item() + DMT_loss_r_eye.item()))
            DMT_face.append(head_pixels/skin_pixels * DMT_loss_skin.item())
            DMT_lips.append(head_pixels/lip_pixels * DMT_loss_lip.item())
            DMT_total.append(DMT_tot.item())
            #DMT_total2.append(DMT_tot2.item())
            BG_eyes.append(head_pixels/eye_pixels * (BG_loss_l_eye.item() + BG_loss_r_eye.item()))
            BG_face.append(head_pixels/skin_pixels * BG_loss_skin.item())
            BG_lips.append(head_pixels/lip_pixels * BG_loss_lip.item())
            BG_total.append(BG_tot.item())
            #BG_total2.append(BG_tot2.item())


        print('----------------')

        print('Eyes', np.mean(our_eyes), np.mean(BG_eyes), np.mean(DMT_eyes))
        print('Face', np.mean(our_face), np.mean(BG_face), np.mean(DMT_face))
        print('Lips', np.mean(our_lips), np.mean(BG_lips), np.mean(DMT_lips))
        print('Total', np.mean(our_total), np.mean(BG_total), np.mean(DMT_total))

        #dict = {'Left Eye': eye_l, 'Right Eye': eye_r, 'Face': face, 'Lips': lips, 'Sum': total}

        #with open('output.csv', 'w') as output:
        #    writer = csv.writer(output)
        #    for key, value in dict.items():
        #        writer.writerow([key, value])

solver = Test()
solver.run()