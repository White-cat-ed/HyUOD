
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.util.shape import view_as_windows
import argparse
import time



def dark_channel(img):
    h, w = img.shape[:2]
    if max(h, w) >= 3000:  
        win_size = 15
    elif max(h, w) >= 1080:  
        win_size = 7
    else:  
        win_size = 3
    b, g, r = cv2.split(img)
    min_img = cv2.min(g, b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win_size, win_size))
    dc_img = cv2.erode(min_img,kernel)
    return dc_img

def get_atmo(img, percent = 0.001):
    mean_perpix = np.mean(img, axis = 2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)

def get_trans(img, atom, w = 0.95):
    x = img / atom
    t = 1 - w * dark_channel(x)
    return t

def guided_filter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    #1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    #2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    #3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    #4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    #5
    q = mean_a * i + mean_b
    return q

def calculate_eta_ratios(t_b, a, lambda_r=700, lambda_g=550, lambda_b=450):
    numerator_r = (-0.00113 * lambda_r + 1.62517) * a[0]#A_b
    denominator_r = (-0.00113 * lambda_b + 1.62517) * a[2]#A_r
    eta_r_over_eta_b = numerator_r / denominator_r
    
    numerator_g = (-0.00113 * lambda_g + 1.62517) * a[0]#A_b
    denominator_g = (-0.00113 * lambda_b + 1.62517) * a[1]#A_g
    eta_g_over_eta_b = numerator_g / denominator_g
    t_r = t_b ** eta_r_over_eta_b
    t_g = t_b ** eta_g_over_eta_b
    
    merged_image = np.stack([t_b, t_g, t_r], axis=-1)
    return merged_image

def calculate_airlight(image, window_size=(25, 25)):

    h, w = image.shape[:2]
    win_h, win_w = window_size

    dark_ = dark_channel(image)

    pixels = []
    for y in range(0, h - win_h + 1, win_h):
        for x in range(0, w - win_w + 1, win_w):

            window = dark_[y:y+win_h, x:x+win_w]

            center_y = y + win_h // 2
            center_x = x + win_w // 2
            pixels.append((window.min(), (center_y, center_x)))
    
    if not pixels:
        return (0.75, 0.75, 0.75)
    
    pixels.sort(reverse=True, key=lambda x: x[0])
    
    top_n = max(int(len(pixels) * 0.01), 1)
    top_coords = [coord for (_, coord) in pixels[:top_n]]
    
    top_rgb_values = []
    for y, x in top_coords:
        if 0 <= y < h and 0 <= x < w:
            top_rgb_values.append(image[y, x])
    
    if not top_rgb_values:
        print("warning")
        return (0.75, 0.75, 0.75)
    
    top_rgb_values = np.array(top_rgb_values)
    airlight = np.mean(top_rgb_values, axis=0)
    
    return tuple(airlight)
        
def dehaze_image(image_path, t_save_path, a_save_path,img_save_path=None):
    im = cv2.imread(image_path)
    # ratio = 640/im.shape[1]
    # im = cv2.resize(im,(640,int(im.shape[0]*ratio)))
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
    atom = calculate_airlight(img)
    trans = get_trans(img, atom)
    trans = np.clip(trans, a_min=0.1, a_max=0.90)
    trans = calculate_eta_ratios(trans, atom)
    t_save = os.path.join(t_save_path, os.path.basename(image_path))
    a_save = os.path.join(a_save_path, os.path.basename(image_path))
    atom = np.full((im.shape[0], im.shape[1], 3), np.multiply(atom, 255), dtype=np.uint8)
    if img_save_path != None:
        img_save = os.path.join(img_save_path, os.path.basename(image_path))
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)  
        stro_img_save = os.path.join(img_save_path,name+'_strong'+ext)
        stro_trans = np.power(trans, 0.5)
        stro_img = stro_trans * im +(1-stro_trans) * atom
        
        fake_stro_img_save = os.path.join(img_save_path,name+'_fakestrong'+ext)
        fake_stro_trans = np.mean(stro_trans)
        fake_stro_img = fake_stro_trans * im +(1-fake_stro_trans) * atom

        cv2.imwrite(img_save, im)
        cv2.imwrite(stro_img_save, stro_img)
        cv2.imwrite(fake_stro_img_save, fake_stro_img)
    cv2.imwrite(t_save, trans * 255)
    cv2.imwrite(a_save, atom)
        
def dehaze_V2(originPath, t_save_path, a_save_path):
    '''originaPath:文件夹的路径，图片上一级
       savePath:同理'''
    image_paths = [os.path.join(originPath, image_name) for image_name in os.listdir(originPath)]
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(dehaze_image, image_path, t_save_path, a_save_path) for image_path in image_paths]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()


if __name__ == "__main__":
    # 1. Initialize the command-line argument parser
    parser = argparse.ArgumentParser(description="Script to batch process dehazing for train and test datasets.")
    
    # 2. Add required command-line arguments
    # First argument: Main input directory for images
    parser.add_argument("input_dir", type=str, help="Main input path for images, e.g., /opt/data/private/UOD/DUO/images")
    # Second argument: Main output directory
    parser.add_argument("output_dir", type=str, help="Main output path, e.g., /opt/data/private/UOD/DUO/")
    
    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Define the list of folder splits to process in one go
    splits = ["test", "train"]

    for split in splits:
        # Construct the corresponding paths
        # e.g., /opt/data/private/UOD/DUO/images/test
        img_path = os.path.join(args.input_dir, split) 
        
        # e.g., /opt/data/private/UOD/DUO/t/test
        out_t_path = os.path.join(args.output_dir, "t", split)
        
        # e.g., /opt/data/private/UOD/DUO/a/test
        out_a_path = os.path.join(args.output_dir, "a", split)
        
        # Automatically create output directories (if they don't exist) to prevent errors
        os.makedirs(out_t_path, exist_ok=True)
        os.makedirs(out_a_path, exist_ok=True)
        
        print(f"Processing '{split}' data...")
        print(f" -> Input image path: {img_path}")
        print(f" -> Output T path: {out_t_path}")
        print(f" -> Output A path: {out_a_path}")
        
        # Call your processing function
        dehaze_V2(img_path, out_t_path, out_a_path)
        
    print("All data processing complete!")
