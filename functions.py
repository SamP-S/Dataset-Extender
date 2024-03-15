import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random

# image and file operations
def load_img(path, colourspace=cv2.IMREAD_COLOR):
    """
    Load image from file
    cv2.IMREAD_COLOR (BGR)
    cv2.IMREAD_GRAYSCALE (G)
    cv2.IMREAD_UNCHANGED (BGR)
    """
    return cv2.imread(path, colourspace)

def save_img(filepath, img):
    return cv2.imwrite(filepath, img)

def load_dir(path):
    imgs = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        imgs.append(load_img(img_path))
    return imgs

def display_image(img):
    plt.imshow(img)
    plt.show()

# colour operations and spaces
def colour_mask(img, lower_colour, upper_colour):
    mask = cv2.inRange(img, lower_colour, upper_colour)
    masked = cv2.bitwise_and(img,img, mask=mask)
    return masked

def colour_filter(img, lower_colour, upper_colour):
    safe = img.copy()
    mask = cv2.inRange(safe, lower_colour, upper_colour)
    masked = cv2.bitwise_and(safe,safe, mask=mask)
    result = safe - masked
    return result

def remove_colour(img, r, g, b, a):
    """
    Set specific colour to transparent
    """
    result = np.copy(img)
    cells = np.where(np.all(img == [b, g, r, a], axis=-1))
    result[cells[0], cells[1], :] = [b, g, r, 0]
    return result

def remove_colour_range(img, c1, c2):
    """
    Set specific colour range to transparent
    """
    mask = cv2.inRange(img, c1, c2) # create the Mask
    mask = 255 - mask  # inverse mask
    return cv2.bitwise_and(img, img, mask=mask)

def colour_histogram(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def brightness(img, factor):
    result = img.copy()
    result = result + np.uint8(factor)
    return result

def contrast(img, factor):
    result = img.copy()
    result = result * factor
    return result.astype(dtype=np.uint8)

# image transforms
def rotate(img, rot):
    """
    Rotates on multiples of 90 degrees
    cv2.ROTATE_90_CLOCKWISE
    cv2.ROTATE_180
    cv2.ROTATE_90_COUNTERCLOCKWISE
    """
    return cv2.rotate(img, rot)

def uniform_scale(img, factor, interp=cv2.INTER_NEAREST):
    """
    Scales the image maintaining aspect ratio
    cv2.INTER_NEAREST
    cv2.INTER_LINEAR
    cv2.INTER_CUBIC
    cv2.INTER_AREA
    """
    scaled_resolution = (img.shape[0] * factor, img.shape[1] * factor)
    return cv2.resize(img, scaled_resolution, interpolation=interp)

def scale(img, x_factor, y_factor, interp=cv2.INTER_NEAREST):
    """
    Scales the image using seperate scaling factors for each dimension
    cv2.INTER_NEAREST
    cv2.INTER_LINEAR
    cv2.INTER_CUBIC
    cv2.INTER_AREA
    """
    scaled_resolution = (int(img.shape[0] * x_factor), int(img.shape[1] * y_factor))
    return cv2.resize(img, scaled_resolution, interpolation=interp)

def mix_images(img, img2, ratio=0.2):
    """
    Mix images using linear ratio
    """
    return (img * ratio).astype(np.uint8) + (img2 * (1 - ratio)).astype(np.uint8)

def insert_image(img, img2, x, y):
    """
    Insert image at coordinates (no mixing/blending)
    """
    result = np.copy(img)
    img_subset = img[y: y + img2.shape[1], x: x + img2.shape[0]]
    result[y: y + img2.shape[1], x: x + img2.shape[0]] = blend_images(img_subset, img2)
    return result

def random_sub_image(img, w, h):
    max_x = img.shape[0] - w
    max_y = img.shape[1] - h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return img[x:x+w, y:y+h]

def random_insert_image(img, img2):
    """
    Insert image at random coordinates
    """
    x = random.randint(0, img.shape[0] - img2.shape[0])
    y = random.randint(0, img.shape[1] - img2.shape[1])
    result = np.copy(img)
    img_subset = img[x: x + img2.shape[0], y: y + img2.shape[1]]
    result[x: x + img2.shape[0], y: y + img2.shape[1]] = blend_images(img_subset, img2)
    return result

def blend_images(img, img2):
    """
    Blend image into another using transparency 
    """
    b1, g1, r1, a1 = cv2.split(img)
    b2, g2, r2, a2 = cv2.split(img2)
    
    a1 = 255 - a2
    
    b = (a1 / 255) * b1 + (a2 / 255) * b2
    g = (a1 / 255) * g1 + (a2 / 255) * g2
    r = (a1 / 255) * r1 + (a2 / 255) * r2
    a = np.maximum(a1, a2)

    b = np.clip(b, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    r = np.clip(r, 0, 255).astype(np.uint8)
    a = np.clip(a, 0, 255).astype(np.uint8)

    blend = cv2.merge([b, g, r, a])
    return blend

# noise
def gaussian_noise(img, strength=1, mean=0, std=20):
    """
    Adds gaussian noise to image according to parameters
    """
    gauss = np.random.normal(mean, std, img.shape)
    noisy = img + gauss * strength
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)
    return noisy

def salt_pepper_noise(img, ratio=0.5, freq=0.05, b_w=False):
    """
    Adds salt and pepper noise ot image
    ratio of salt to pepper (higher ration -> more salt)
    single channel or B & W
    """
    noisy = np.copy(img)
    amount = int(freq * img.shape[0] * img.shape[1] * img.shape[2])
    salt = int(ratio * amount)

    for i in range(amount):
        coord = [np.random.randint(0, i) for i in img.shape]
        if i <= salt:
            if b_w:
                noisy[coord[0]][coord[1]] = [255 for i in range(img.shape[2])]
            else:
                noisy[coord[0]][coord[1]][coord[2]] = 255
        else:
            if b_w:
                noisy[coord[0]][coord[1]] = [0 for i in range(img.shape[2])]
            else:
                noisy[coord[0]][coord[1]][coord[2]] = 0
    return noisy

def poisson_noise(img):
    """
    Add poisson noise to image
    """
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return noisy

def speckle_noise(img):
    """
    Add speckle noise
    """
    row,col,ch = img.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = img + img * gauss
    return noisy

if __name__ == "__main__":
    cwd = os.getcwd()
    bricks_dir = os.path.join(cwd, "bricks")
    bg_dir = os.path.join(cwd, "backgrounds")

    brick_imgs = load_dir(bricks_dir)
    bg_imgs = load_dir(bg_dir)

    test_img = brick_imgs[0]
    test_img2 = brick_imgs[1]

    # display_image(test_img)
    # display_image(test_img2)

    # display_image(rotate(test_img, cv2.ROTATE_90_CLOCKWISE))
    # display_image(uniform_scale(test_img, 2, cv2.INTER_NEAREST))
    # display_image(scale(test_img, 2, 0.5, cv2.INTER_NEAREST))

    # display_image(gaussian_noise(test_img, 2))
    # display_image(salt_pepper_noise(test_img))
    # display_image(salt_pepper_noise(test_img, b_w=True))

    # display_image(mix_images(test_img, test_img2))
    # scaled_test2_img = scale(test_img2, 0.5, 0.5)
    # display_image(insert_image(test_img, scaled_test2_img, 100, 100))

    # colour_histogram(test_img)

    # bg_remove = remove_colour(test_img, 70, 70, 70, 255)
    # bg_remove = remove_colour(bg_remove, 71, 71, 71, 255)
    # display_image(bg_remove)

    # bg_rem = remove_colour_range(test_img, (70, 70, 70, 255), (72, 72, 72, 255))
    # display_image(blend_images(bg_imgs[0], bg_rem, 100, 100))

    display_image(poisson_noise(test_img))
    display_image(speckle_noise(test_img))

