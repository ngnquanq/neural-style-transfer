import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import model
import utils
import argparse
import os
from os import path

nst = model.nst_model
loader = utils.loader
imshow = utils.imshow

dir_path = os.path.dirname(os.path.realpath(__file__))
def combine_image(content_image: str, style_image: str):
    #imshow = utils.imshow
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_images = r"C:\Users\84898\NST\images"
    content_image_path = path_to_images + fr'\{content_image}'
    style_image_path = path_to_images + fr'\{style_image}'
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img', type=str, default = fr'{content_image_path}')
    parser.add_argument('--style_img', type=str, default = fr'{style_image_path}')
    parser.add_argument('--size', type=int, default = 128, help='if you want to get more clear pictures, increase the size')
    parser.add_argument('--steps', type=int, default = 300)
    parser.add_argument('--c_weight', type=int, default = 1, help='weighting factor for content reconstruction')
    parser.add_argument('--s_weight', type=int, default = 100000, help='weighting factor for style reconstruction')

    args = parser.parse_args()
    print(args)
    output = main(args)

    plt.figure()
    imshow(output, title = 'Output Image')
    plt.pause(3)
    plt.show()

def main(args):
    content_img = args.content_img
    style_img = args.style_img
    size = args.size
    steps = args.steps
    c_weight = args.c_weight
    s_weight = args.s_weight

    content_img, style_img = loader(content_img, style_img, size = size)
    input_img = content_img.clone() # just noise array is fine


    model, style_losses, content_losses  = nst(content_img, style_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    step = [0]
    while step[0] <= steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            output = model(input_img)

            cl = 0
            sl = 0

            for c_loss in content_losses:
                cl += c_loss.loss * c_weight
            for s_loss in style_losses:
                sl += s_loss.loss * s_weight

            loss = cl + sl
            loss.backward()

            if step[0] % 50 == 0:
                print('Step : {}'. format(step))
                print('Style Loss : {:3f} Content Loss: {:3f}'.format(
                    sl.item(), cl.item()))

            step[0] += 1

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0,1)
    return input_img

    imshow(content_img, title = 'Input image')
    plt.show()

def run():
    content_image = str(input(print("Type in the content image:\n")))
    style_image = str(input(print("Type in the style image:\n")))
    combine_image(content_image=content_image, style_image=style_image)

if __name__ == "__main__":
    print("\nPlease notice that only accept .jpg file")
    content = input("Type in the content image:\n")
    style = input("Type in the style image\n")

    combine_image(content, style)