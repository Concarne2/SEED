import os
import json
import numpy as np
import scipy as sp
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer


# EffNet evaluation code brought from MindEyeV2
def eval_effnet(recon_img_dir, gt_img_dir):

    recon_paths = sorted(
        [os.path.join(recon_img_dir, name) for name in os.listdir(recon_img_dir)],
        key=lambda x: os.path.basename(x),
    )
    gt_paths = sorted(
        [os.path.join(gt_img_dir, name) for name in os.listdir(gt_img_dir)],
        key=lambda x: os.path.basename(x),
    )

    def load_images(paths):
        images = []
        for path in paths:
            with Image.open(path) as img:
                array = np.array(img.convert("RGB"), copy=True)
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(images, dim=0).float().div_(255.0)

    all_recons = load_images(recon_paths)
    all_images = load_images(gt_paths)

    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                        return_nodes=['avgpool'])
    eff_model.eval().requires_grad_(False)

    # see weights.transforms()
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    gt = eff_model(preprocess(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()

    effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))])

    return effnet

def eval_object_f1(recon_detection_dir, gt_detection_dir, thresholds = None, detector_area_threshold = 50, verbose = False):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
    
    img_list = os.listdir(os.path.join(recon_detection_dir, "preds"))
    img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))

    recall_list_all = []
    precision_list_all = []

    for img in img_list:
        recall_list_img = []
        precision_list_img = []

        for threshold in thresholds:
            # getting detected categories of the reconstructed image
            json_file = os.path.join(recon_detection_dir, "preds", img)

            with open(json_file) as f:
                data = json.load(f)
            recon_labels = data['labels']
            recon_scores = data['scores']
            recon_bboxes = data['bboxes']

            recon_cats = []

            for i in range(len(recon_labels)):
                if recon_scores[i] < threshold: break

                area = (recon_bboxes[i][2] - recon_bboxes[i][0]) * (recon_bboxes[i][3] - recon_bboxes[i][1])
                if area < detector_area_threshold: continue
                
                if recon_labels[i] not in recon_cats:
                    recon_cats.append(recon_labels[i])
            
            # getting detected categories of the ground truth image
            json_file = os.path.join(gt_detection_dir, "preds", img)
            with open(json_file) as f:
                data = json.load(f)
            gt_labels = data['labels']
            gt_scores = data['scores']
            gt_bboxes = data['bboxes']

            gt_cats = []
            
            for i in range(len(gt_labels)):
                if gt_scores[i] < threshold: break

                area = (gt_bboxes[i][2] - gt_bboxes[i][0]) * (gt_bboxes[i][3] - gt_bboxes[i][1])
                if area < detector_area_threshold: continue

                if gt_labels[i] not in gt_cats:
                    gt_cats.append(gt_labels[i])
            
            # Calculating recall
            if len(gt_cats) > 0:
                recall = 0

                for cat in gt_cats:
                    if cat in recon_cats:
                        recall += 1
                
                recall /= len(gt_cats)
                recall_list_img.append(recall)

            # Calculating precision
            if len(recon_cats) > 0:
                precision = 0

                for cat in recon_cats:
                    if cat in gt_cats:
                        precision += 1
                
                precision /= len(recon_cats)
                precision_list_img.append(precision)

            # If no categories are detected, move on to the next image
            if len(gt_cats) == 0 and len(recon_cats) == 0:
                break
        
        if len(recall_list_img) == 0:
            if verbose:
                print("No GT categories detected for image", img)
            recall_list_all.append(1)
        else:
            average_recall = sum(recall_list_img) / len(recall_list_img)
            recall_list_all.append(average_recall)
        
        if len(precision_list_img) == 0:
            if verbose:
                print("No reconstructed categories detected for image", img)
            precision_list_all.append(1)
        else:
            average_precision = sum(precision_list_img) / len(precision_list_img)
            precision_list_all.append(average_precision)

    total_recall = sum(recall_list_all) / len(recall_list_all)
    total_precision = sum(precision_list_all) / len(precision_list_all)

    if verbose:
        print("Total Recall: ", "{:.3f}".format(total_recall))
        print("Total Precision: ", "{:.3f}".format(total_precision))
    
    f1_list_all = []
    for i in range(len(precision_list_all)):
        if precision_list_all[i] == 0 and recall_list_all[i] == 0:
            f1_list_all.append(0)
        else:
            f1_list_all.append(2*(precision_list_all[i]*recall_list_all[i])/(precision_list_all[i]+recall_list_all[i]))
    
    return np.array(f1_list_all)

def eval_cap_sim(recon_img_dir, gt_img_dir):

    recon_paths = sorted(
        [os.path.join(recon_img_dir, name) for name in os.listdir(recon_img_dir)],
        key=lambda x: os.path.basename(x),
    )
    gt_paths = sorted(
        [os.path.join(gt_img_dir, name) for name in os.listdir(gt_img_dir)],
        key=lambda x: os.path.basename(x),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    caption_model_dtype = torch.float32
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    caption_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/git-large-coco",
        torch_dtype=caption_model_dtype,
    ).to(device)
    caption_model.eval()

    def _generate_captions(paths, batch_size=16):
        captions = {}
        for start_idx in range(0, len(paths), batch_size):
            batch_paths = paths[start_idx:start_idx + batch_size]
            batch_ids = [os.path.basename(path) for path in batch_paths]
            batch_images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    batch_images.append(img.convert("RGB"))

            pixel_values = processor(images=batch_images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            with torch.inference_mode():
                generated_ids = caption_model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=50,
                )

            batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for image_id, caption in zip(batch_ids, batch_captions):
                captions[image_id] = caption
        return captions

    recon_captions = _generate_captions(recon_paths)
    gt_captions = _generate_captions(gt_paths)

    common_image_ids = sorted(set(gt_captions.keys()) & set(recon_captions.keys()))
    if len(common_image_ids) == 0:
        raise ValueError("No matching image filenames found between GT and reconstruction inputs.")

    gt_texts = [gt_captions[image_id] for image_id in common_image_ids]
    recon_texts = [recon_captions[image_id] for image_id in common_image_ids]

    text_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    gt_embeddings = text_model.encode(
        gt_texts,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    recon_embeddings = text_model.encode(
        recon_texts,
        batch_size=128,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    cossims = np.sum(gt_embeddings * recon_embeddings, axis=1)
    return recon_captions, gt_captions, cossims
