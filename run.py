# Standard library
import csv
import os
import warnings
from contextlib import nullcontext

# Third-party
import nd2
import numpy as np
import skimage
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.amp import GradScaler
from tqdm import tqdm

from scipy import stats
from scipy.ndimage import (
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label,
)

from skimage.color import rgb2gray
from skimage.draw import polygon
from skimage.feature import peak_local_max
from skimage.filters import threshold_triangle
from skimage.measure import regionprops
from skimage.segmentation import watershed

from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor

# Local
from find_device_edges import get_image_mask, pad_with_mask, preprocess_raw_image_fast
from tune_boundary_extraction import analyze_growth, create_rectangular_mask
from utils import load_training_filenames, preprocess_0_1
from xlstm.twoDUVixLSTM import UVixLSTM

def twice_smooth_and_threshold(img):
    gfilt = gaussian_filter(img, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    gfilt = gaussian_filter(gfilt, 1)
    gfilt = (gfilt > 0.4).astype(np.float32)
    return gfilt.astype(np.uint8)

def generate_edge_mask(image_shape, rim_pixel_width, device):
    edge_mask = torch.zeros(image_shape).to(device)
    max_index = image_shape[0]
    main_start = rim_pixel_width
    main_end = max_index - rim_pixel_width
    edge_mask[main_start:main_end, 
            main_start:main_end] = 1
    edge_mask = edge_mask.float()
    edge_mask = edge_mask.unsqueeze(0).repeat(1, 1, 1, 1).to(device)
    return edge_mask
    

def random_rotate_batch_2d(images, labels):
    batch_size = images.size(0)
    
    # Randomly choose rotation angles (0, 90, 180, 270 degrees)
    angles = torch.randint(0, 4, (batch_size,), device=images.device)
    
    rotated_images = []
    rotated_labels = []
    
    for i in range(batch_size):
        # Both images and labels are 3D tensors, so use dims=(1,2)
        rotated_image = torch.rot90(images[i], k=int(angles[i]), dims=(0, 1))
        rotated_label = torch.rot90(labels[i], k=int(angles[i]), dims=(0, 1))
            
        rotated_images.append(rotated_image)
        rotated_labels.append(rotated_label)
    
    return torch.stack(rotated_images), torch.stack(rotated_labels)


def gaussian_noise_augmentation(input_images):
    """
    Adds Gaussian noise to a batch of images with probability 0.2,
    ensuring output stays in [0, 1].
    
    Args:
        input_images: torch.Tensor of shape (B, C, H, W), values in [0,1]
    
    Returns:
        torch.Tensor of same shape, augmented images
    """
    gaussprob = np.random.uniform(0, 1)
    if gaussprob > 0.8:
        noise = torch.normal(mean=0.0, std=0.3, size=input_images.shape, device=input_images.device)
        input_images = input_images + noise
        input_images = torch.clamp(input_images, 0.0, 1.0)
    return input_images

def motion_blur_augmentation(input_images, device):
    gaussprob0 = np.random.uniform(0, 1)
    if gaussprob0 > 0.8:
        #kernel = np.random.choice(possible_kernels, 3)
        kernels = (3, 3)
        #kernels = tuple(kernel)
        input_images = torch.stack([motion_blur_2d_torch(img, np.random.randint(1, 60), kernels, device) for img in input_images])
    return input_images

def watershed_refined(prediction, min_distance = 3, centre_threshold = 0.28, background_threshold = 0.1):
    inference = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    maxima_locs = peak_local_max(inference, min_distance=min_distance, threshold_rel = centre_threshold, p_norm=2.0)
    maxima = np.zeros_like(inference).astype(int)
    maxima[maxima_locs[:, 0], maxima_locs[:, 1]] = 1
    labeled_array, _ = label(maxima)
    background_image = (inference > background_threshold).astype(int)
    wts = watershed(-inference, labeled_array, mask = background_image) # -distance
    slices = find_objects(wts)
    wts_refined = np.zeros_like(wts)
    idx = 1
    for i, slice_tuple in enumerate(tqdm(slices), start=1):
        if slice_tuple is not None:
            local_locs = np.array(np.where(wts[slice_tuple] == i))
            global_locs = np.stack(local_locs).T + np.array([s.start for s in slice_tuple])
            #distribution_cell_sizes.append(len(global_locs))
            if len(global_locs) < 60:
                wts_refined[global_locs[:, 0], global_locs[:, 1]] = idx
                idx += 1
    print("num cells", idx)
    return wts_refined

from skimage.morphology import disk


def motion_blur_2d_torch(image, angle, kernel_size, device):
    """
    Apply 2D motion blur to a 2D tensor image using PyTorch.

    Parameters:
    - image (torch.Tensor): Input 2D image tensor (dtype should be float for accurate blurring).
    - angle (float): Angle of motion blur in degrees.
    - kernel_size (int): Size of the motion blur kernel.
    - device (torch.device): Device to perform computations on.

    Returns:
    - blurred_image (torch.Tensor): Blurred 2D image tensor.
    """
    # Convert angle to radians
    angle_rad = torch.tensor(np.radians(angle), dtype=torch.float32)
    
    # Generate 2D motion blur kernel
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    
    # Calculate center point of kernel
    center = (kernel_size - 1) / 2
    offsets = torch.linspace(-center, center, steps=kernel_size, dtype=torch.float32)
    x, y = torch.meshgrid(offsets, offsets, indexing='ij')
    
    # Compute motion blur direction vector
    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)
    direction = cos_theta * x + sin_theta * y

    # Normalize kernel
    kernel += torch.exp(-torch.abs(direction))
    kernel /= kernel.sum()
    kernel = kernel.to(device)
    
    # Apply motion blur using 2D convolution
    padding = kernel_size // 2
    blurred_image = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding)
    
    return blurred_image.squeeze(0).squeeze(0)

def ensure_grayscale(img):
  if img.ndim == 2:
        return img  # already grayscale
  elif img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Drop alpha channel
        if img.shape[2] == 3:
            return rgb2gray(img)
  raise ValueError(f"Unsupported image shape: {img.shape}")

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import os
    print(os.getcwd())

    # replace these with your own paths
    train_path = "/home/laurids/Desktop/microclot_project/eval_folder/train"
    train_val_path = "/home/laurids/Desktop/microclot_project/eval_folder/val"
    loading_folder = "/media/laurids/Elements/microclot_images/all_images"
    neural_predictions_path = "/media/laurids/Elements/microclot_images/just_testing"

    img, heat = load_training_filenames(train_path, label_format=".tif")
    bpath = train_path
    images = [skimage.io.imread(os.path.join(bpath, f)) for f in img]
    heat = [skimage.io.imread(os.path.join(bpath, f)) for f in heat]

    val_imgs, val_heats = load_training_filenames(train_val_path, label_format=".tif")
    val_imgs = [skimage.io.imread(os.path.join(train_val_path, f)) for f in val_imgs]
    val_heats = [skimage.io.imread(os.path.join(train_val_path, f)) for f in val_heats]


    images = [np.pad(img, ((400, 400), (400, 400)), mode='reflect') for img in images]
    heat = [np.pad(img, ((400, 400), (400, 400)), mode='constant', constant_values=-100) for img in heat]
    val_imgs = [np.pad(img, ((400, 400), (400, 400)), mode='reflect') for img in val_imgs]
    val_heats = [np.pad(img, ((400, 400), (400, 400)), mode='constant', constant_values=-100) for img in val_heats]
    images = [preprocess_0_1(img, low_clip=1.0, high_clip=99.6, clip=True) for img in images]
    val_imgs = [preprocess_0_1(img, low_clip=1.0, high_clip=99.6, clip=True) for img in val_imgs]

    thresholded_labs = []
    for hh in heat:
        thresh = hh > 18
        thresh_lab, _ = label(thresh)
        thresholded_labs.append(thresh_lab)

    thresholded_val_labs = []
    for hh in val_heats:
        thresh = hh > 18
        thresh_lab, _ = label(thresh)
        thresholded_val_labs.append(thresh_lab)


    list_of_label_dicts = []
    for img in thresholded_labs:
        image_label_map = {}
        
        # 1. Get slices for all labels at once
        # find_objects returns a list where index 0 is label 1, index 1 is label 2, etc.
        slices = find_objects(img)
        
        for i, SL in enumerate(slices):
            label_idx = i + 1  # find_objects starts at label 1
            
            if SL is not None:
                # 2. Extract the local patch for this label
                patch = img[SL]
                
                # 3. Find coordinates where the pixel equals our current label
                # These coords are LOCAL to the patch
                local_coords = np.argwhere(patch == label_idx)
                
                if local_coords.shape[0] > 0:
                    # 4. Calculate local median
                    local_median = np.median(local_coords, axis=0)
                    
                    # 5. Convert Local -> Global
                    # Add the 'start' of the slice to the local median
                    global_y = int(local_median[0] + SL[0].start)
                    global_x = int(local_median[1] + SL[1].start)
                    image_label_map[label_idx] = (global_y, global_x)
        list_of_label_dicts.append(image_label_map)


    list_of_label_dicts_val = []
    for img in thresholded_val_labs:
        image_label_map = {}
        
        # 1. Get slices for all labels at once
        # find_objects returns a list where index 0 is label 1, index 1 is label 2, etc.
        slices = find_objects(img)
        
        for i, SL in enumerate(slices):
            label_idx = i + 1  # find_objects starts at label 1
            
            if SL is not None:
                # 2. Extract the local patch for this label
                patch = img[SL]
                
                # 3. Find coordinates where the pixel equals our current label
                # These coords are LOCAL to the patch
                local_coords = np.argwhere(patch == label_idx)
                
                if local_coords.shape[0] > 0:
                    # 4. Calculate local median
                    local_median = np.median(local_coords, axis=0)
                    
                    # 5. Convert Local -> Global
                    # Add the 'start' of the slice to the local median
                    global_y = int(local_median[0] + SL[0].start)
                    global_x = int(local_median[1] + SL[1].start)
                    
                    image_label_map[label_idx] = (global_y, global_x)
                    
        list_of_label_dicts_val.append(image_label_map)

    rng = np.random.default_rng(seed=42)
    input_images = images
    train_labels_intensity = heat

    model = UVixLSTM(class_num = 1, img_dim = 512, in_channels=1).to(device)
    base_lr = 0.000001 
    max_lr = 0.01 
    step_size_up = 400 
    step_size_down = 250
    training = False

    if training:

        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        loss_fn = torch.nn.MSELoss(reduction="none")
        num_images = len(input_images)
        num_mini = 1
        scaler = GradScaler("cuda")
        preloaded_images = [torch.from_numpy(img).float() for img in images]
        preloaded_labels = [torch.from_numpy(lbl).float() for lbl in train_labels_intensity]

        # Keep label dicts as-is
        preloaded_label_dicts = list_of_label_dicts.copy()

        num_images = len(preloaded_images)
        num_mini = 1
        scaler = GradScaler("cuda")
        maximum_epochs = 60_000
        best_val_loss = np.inf

        val_crops_imgs = []
        val_crops_labels = []

        for val_index, ll in enumerate(list_of_label_dicts_val):
            val_img_full = val_imgs[val_index]
            val_heat_full = val_heats[val_index]

            for key in ll.keys():
                val_img_median = np.array(ll[key])

                # Crop patch with boundary clipping
                x0 = max(0, val_img_median[0]-256)
                x1 = min(val_img_full.shape[0], val_img_median[0]+256)
                y0 = max(0, val_img_median[1]-256)
                y1 = min(val_img_full.shape[1], val_img_median[1]+256)

                # Store as NumPy arrays (CPU)
                val_crops_imgs.append(val_img_full[x0:x1, y0:y1])
                val_crops_labels.append(val_heat_full[x0:x1, y0:y1])

        # --- Training loop ---
        for jj in tqdm(range(maximum_epochs)):
            model.train()
            optimizer.zero_grad()

            selected_index = np.random.randint(0, num_images)
            input_img = preloaded_images[selected_index]        # torch tensor, CPU
            label_img = preloaded_labels[selected_index]
            label_dict = preloaded_label_dicts[selected_index]

            objects_to_sample = len(label_dict.keys())
            selected_location = np.random.randint(1, objects_to_sample + 1)
            median_location = np.array(label_dict[selected_location])
            random_jitter = np.random.randint(-50, 51, size=(2,))
            median_location += random_jitter

            x0 = max(0, median_location[0]-256)
            x1 = min(input_img.shape[0], median_location[0]+256)
            y0 = max(0, median_location[1]-256)
            y1 = min(input_img.shape[1], median_location[1]+256)

            input_patch = input_img[x0:x1, y0:y1].unsqueeze(0)       # add batch dim
            label_patch = label_img[x0:x1, y0:y1].unsqueeze(0)

            input_images = input_patch.to(device)
            labels_inten = label_patch.to(device)

            edge_mask = generate_edge_mask((input_images[0].shape[0], input_images[0].shape[1]), 10, device)
            input_images, labels_inten = random_rotate_batch_2d(input_images, labels_inten)
            gaussprob0 = np.random.uniform(0, 1)
            if gaussprob0 > 0.8:
                input_images = torch.stack([motion_blur_2d_torch(img, np.random.randint(1, 60), 3, device) for img in input_images])
            input_images = gaussian_noise_augmentation(input_images)
            gaussprob = np.random.uniform(0, 1)
            if torch.isnan(input_images).any():
                import pdb; pdb.set_trace()

            # Forward pass
            if device.type == 'cuda':
                with autocast("cuda"):
                    output = model(input_images.unsqueeze(1)) 
                    output = output.squeeze(1)
                    mask = torch.where(labels_inten == -100, 0.0, 1.0)
                    loss1 = loss_fn(output, labels_inten) * mask
                    loss1 = torch.mean(loss1)
                    loss = loss1 

            # Backward pass
            if device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                #scheduler.step()

            if jj % 500 == 0 and jj > 200:

                model.eval()
                loss_sum = []

                for val_img_np, val_label_np in zip(val_crops_imgs, val_crops_labels):

                    # Convert to torch and move to device
                    val_images_torch = torch.from_numpy(val_img_np).unsqueeze(0).float().to(device)
                    val_labels_intensity_torch = torch.from_numpy(val_label_np).unsqueeze(0).float().to(device)

                    # Generate edge mask
                    edge_mask_val = generate_edge_mask((val_img_np.shape[0], val_img_np.shape[1]), 10, device)

                    # Forward + compute loss
                    with torch.no_grad(), autocast("cuda") if device.type == 'cuda' else torch.no_grad():
                        val_output = model(val_images_torch.unsqueeze(1))  # add channel dim
                        val_output = val_output.squeeze(1)
                        mask = torch.where(val_labels_intensity_torch != -100, 1.0, 0.0)
                        val_loss1 = loss_fn(val_output, val_labels_intensity_torch) * mask
                        val_loss1 = torch.mean(val_loss1)
                        loss_sum.append(val_loss1.item())

                mean_val_loss = np.mean(loss_sum)
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    patience_counter = 0
                    best_model = model
                else:
                    patience_counter += 1

                print("epoch", jj, "loss", loss.item(), "mean_val_loss", mean_val_loss, "patience_counter", patience_counter)

        torch.save(best_model.state_dict(), "microclot.pth")
        model = best_model

    else:
        model.load_state_dict(torch.load("microclot.pth"))
    model.eval()

    def center_weighted_array(shape):
        """
        Returns a NumPy array of given 2D shape with:
        - maximum value at the center
        - 0 at the outermost pixel layer
        - decaying with Euclidean distance from the outermost layer
        """
        mask = np.ones(shape, dtype=bool)
        mask[1:-1, 1:-1] = False

        edt = distance_transform_edt(~mask)
        edt[mask] = 0

        max_val = edt.max()
        if max_val > 0:
            edt = edt / max_val
        else:
            edt = np.zeros_like(edt)

        return edt

    def get_autocast(mixed_precision=True):
        if not mixed_precision:
            return nullcontext()

        # New API (PyTorch 2.0+)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", dtype=torch.float16)

        # Old API (PyTorch ≤1.13)
        return torch.cuda.amp.autocast(dtype=torch.float16)

    def sliding_window_loop_2d(
        H, W, keep_size, step_size, image_dim, model,
        low_value, high_value, mixed_precision, image,
        device, tta=False, offset=True,
        patch_based_norm=False, euclidean_feathering=False
    ):
        autocast_context = get_autocast(mixed_precision)

        pad_h = (image_dim[0] - keep_size[0]) // 2
        pad_w = (image_dim[1] - keep_size[1]) // 2

        output_image = np.zeros((H, W), dtype=np.float32)
        count_image = np.zeros_like(output_image, dtype=np.float32)

        step_h, step_w = step_size

        if euclidean_feathering:
            total_weights = np.zeros_like(output_image, dtype=np.float32)
            weight_times_value = np.zeros_like(output_image, dtype=np.float32)
            edt_array = center_weighted_array((image_dim[0], image_dim[1]))

        if offset:
            offset_h = step_h // 2
            offset_w = step_w // 2
        else:
            offset_h = 0
            offset_w = 0

        for h in range(offset_h, H - keep_size[0] + 1, step_h):
            for w in range(offset_w, W - keep_size[1] + 1, step_w):

                h_start = max(0, h - pad_h)
                w_start = max(0, w - pad_w)

                h_end = min(image.shape[0], h_start + image_dim[0])
                w_end = min(image.shape[1], w_start + image_dim[1])

                # Skip incomplete patches
                if (h_end - h_start < image_dim[0] or
                    w_end - w_start < image_dim[1]):
                    continue

                patch = image[h_start:h_end, w_start:w_end]

                if patch_based_norm:
                    patch = preprocess_0_1(patch, low_clip=1.0, high_clip=99.9)

                if tta:
                    avg_patch = np.zeros_like(patch, dtype=np.float32)
                    n_transforms = 0

                    for k in [0, 1, 2, 3]:
                        rotated_patch = np.rot90(patch, k).copy()
                        patch_torch = torch.from_numpy(rotated_patch).float().to(device)

                        if device.type == 'cuda':
                            with torch.no_grad(), autocast_context:
                                pred = model(patch_torch.unsqueeze(0).unsqueeze(0))
                        else:
                            with torch.no_grad():
                                pred = model(patch_torch.unsqueeze(0).unsqueeze(0))

                        pred = pred.detach().cpu().numpy().squeeze()
                        pred = np.rot90(pred, -k)
                        avg_patch += pred
                        n_transforms += 1

                    predicted_patch = avg_patch / n_transforms
                    predicted_patch = np.clip(predicted_patch, low_value, high_value)

                else:
                    patch_torch = torch.from_numpy(patch).float().to(device)
                    if device.type == 'cuda':
                        with torch.no_grad(), autocast_context:
                            predicted_patch = model(patch_torch.unsqueeze(0).unsqueeze(0))
                    else:
                        with torch.no_grad():
                            predicted_patch = model(patch_torch.unsqueeze(0).unsqueeze(0))

                    predicted_patch = predicted_patch.detach().cpu().numpy().squeeze()
                    predicted_patch = np.clip(predicted_patch, low_value, high_value)

                if np.isnan(predicted_patch).any():
                    warnings.warn("NaN detected in predicted_patch, substituting background")
                    predicted_patch[np.isnan(predicted_patch)] = low_value

                if euclidean_feathering:
                    weight_times_value[h_start:h_end, w_start:w_end] += predicted_patch * edt_array
                    total_weights[h_start:h_end, w_start:w_end] += edt_array
                else:
                    actual_pad_h = h - h_start
                    actual_pad_w = w - w_start

                    center_h_start = actual_pad_h
                    center_w_start = actual_pad_w
                    center_h_end = center_h_start + keep_size[0]
                    center_w_end = center_w_start + keep_size[1]

                    center_patch = predicted_patch[
                        center_h_start:center_h_end,
                        center_w_start:center_w_end
                    ]

                    output_image[h:h + keep_size[0],
                                w:w + keep_size[1]] += center_patch

                    count_image[h:h + keep_size[0],
                                w:w + keep_size[1]] += 1

        if euclidean_feathering:
            output_image = np.divide(
                weight_times_value, total_weights,
                out=np.zeros_like(output_image),
                where=total_weights != 0
            )
        else:
            output_image = np.divide(
                output_image, count_image,
                out=np.zeros_like(output_image),
                where=count_image != 0
            )

        return output_image
    
    def compute_padding(D, step, keep_size, input_tile):
        side_padding = 2 * (input_tile - keep_size)
        leftover = D % step
        return leftover + side_padding
    
    keep_size = [256, 256]
    step_size = [256, 256]
    image_dim = [512, 512]
    padding_list = []
    inference_images_torch = []
    inference_image_names = os.listdir(loading_folder)
    for name in tqdm(inference_image_names):

        with nd2.ND2File(os.path.join(loading_folder, name)) as f:
            metadata = f.asarray()

        edge_mask = preprocess_raw_image_fast(metadata)
        first_mask = analyze_growth(edge_mask, False)
        rectangular_mask, edt_to_walls, distance_to_end = create_rectangular_mask(first_mask)
        inference_images = pad_with_mask(metadata, rectangular_mask) # use reflective padding outside of mask
        inference_images = preprocess_0_1(inference_images, low_clip=1.0, high_clip=99.6, clip=True)
        padding_width = [0, 0]
        image_shape = inference_images.shape
        # pad image to guarantee proper boundary handling
        for i, shape in enumerate(image_shape):
            padding_width[i] = padding_width[i] + compute_padding(shape, step_size[i], keep_size[i], image_dim[i])
        padded_values = [(padding_width[i] // 2, padding_width[i] - padding_width[i] // 2) for i in range(len(keep_size))]
        padded_image = np.pad(inference_images, 
                            pad_width=((padded_values[0][0], padded_values[0][1]),
                                        (padded_values[1][0], padded_values[1][1])),
                            mode='reflect')
        inference_images = padded_image
        inference_images = inference_images
        H, W = inference_images.shape
        image_dim = [512, 512]
        padding = padded_values
        res = sliding_window_loop_2d(H,
                                    W,
                                    keep_size,
                                    step_size,
                                    image_dim,
                                    model,
                                    -5.0,
                                    20.0,
                                    True,
                                    inference_images,
                                    device,
                                    False,
                                    False,
                                    False,
                                    True
                                    )
        output = res[padding[0][0]:res.shape[0]-padding[0][1], padding[1][0]:res.shape[1]-padding[1][1]]
        threshold = 670 / 65535
        image_mask = inference_images > threshold
        output = output + 5.0
        output /= 25.0
        image_mask = image_mask[padding[0][0]:res.shape[0]-padding[0][1], padding[1][0]:res.shape[1]-padding[1][1]]
        output *= image_mask
        output*= rectangular_mask
        truncated_name = name.split(".")[0]
        adjusted_name = truncated_name + "_output.tif"
        output_thresholded = twice_smooth_and_threshold((output>0.1).astype(np.float32))
        tifffile.imwrite(os.path.join(neural_predictions_path, adjusted_name), output_thresholded)
        tifffile.imwrite(os.path.join(neural_predictions_path, adjusted_name.replace("_output.tif", "_mask.tif")), rectangular_mask)
        binary_mask = output_thresholded.astype(bool)
        L, n_objects = label(binary_mask)

        # Extract region properties safely
        regions = regionprops(L)
        csv_name = f"{truncated_name}_output_summary.csv"
        csv_file = os.path.join(neural_predictions_path, csv_name)
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            if not regions:
                writer.writerow([-1])
            else:
                header = ["object_id", "area", "elongation", "axis_ratio", "compactness", "wall_dist", "end_dist"]
                writer.writerow(header)
                for i, r in enumerate(regions):
                    area = r.area
                    elongation = r.eccentricity
                    axis_ratio = (r.major_axis_length / r.minor_axis_length 
                                if r.minor_axis_length > 0 else 0)
                    compactness = 4 * np.pi * area / (r.perimeter ** 2) if r.perimeter > 0 else 0
                    # Median distances for this object
                    obj_mask = L == r.label
                    wall_dist = np.median(edt_to_walls[obj_mask])
                    end_dist = np.median(distance_to_end[obj_mask])
                    writer.writerow([i + 1, area, elongation, axis_ratio, compactness, wall_dist, end_dist])
