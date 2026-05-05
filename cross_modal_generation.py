import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
import cv2
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def affine_matrix(scale=1.0, theta=0.0, trans=np.zeros(2), trans_2=np.zeros(2)): 
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * scale # 
    t = np.dot(R, trans) + trans_2 # 
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    print(f'loading the model from {ckpt}, this is the final loading model')
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    center = np.array([256, 256]) 
    img_c = np.array(image.shape[1::-1]) / 2.0
    T = affine_matrix( 
        scale=1, 
        theta=0, 
        trans=-img_c, 
        trans_2=center)
    image = cv2.warpAffine(image, #
        T[:2], 
        dsize=tuple(np.array([512, 512])), 
        flags=cv2.INTER_LINEAR, borderValue=255)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = image[None,None,...]
    image = torch.from_numpy(image)
    return image

# testing loading image and load model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default= "configs/fingerprint-cldm-vq-128-512-contactless-eval.yaml", #
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default= "models/fingerprint_c2cl_512/model_unsw_flat.ckpt", #
        help="path to checkpoint of model",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )

    parser.add_argument(
        "--img-folder",
        type=str,
        nargs="?",
        help="path to the input image folders"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="path to the save imgs",
        default="./example",
    )

    parser.add_argument( # 50 steps
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument( # dete is ok
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}", verbose=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    batch_size = opt.n_samples
    # process the input image
    opt.img_folder = os.path.abspath(opt.img_folder)
    image_paths = os.listdir(opt.img_folder)
    sample_path = os.path.join(opt.outdir, 'contactless_texture') 
    os.makedirs(sample_path, exist_ok=True)
    for image_path in image_paths:
        # assert os.path.isfile(opt.init_img)
        full_image_path = os.path.join(opt.img_folder, image_path)
        init_image = load_img(full_image_path).to(device)
        # b is the view size
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size) # out the 3 views
        # out the ID_emb and 
        cond = dict()
        cond['ID'] = init_image
        sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
        # do not need to set t_enc, it is set for control condition
        precision_scope = autocast  #if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope("Plotting"): # testing
                    # using the control condition
                    shapes = (3, 128, 128)
                    samples,_ = sampler.sample(opt.ddim_steps, batch_size, 
                                            shapes, cond, verbose=False,
                                            eta=opt.ddim_eta,)
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        x_sample = x_sample.squeeze()
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, image_path))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()