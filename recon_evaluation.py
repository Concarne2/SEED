from argparse import ArgumentParser
from seed.metrics import eval_cap_sim, eval_effnet, eval_object_f1
import numpy as np
import os


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Name of the model being evaluated. Used for logging purposes.')
    parser.add_argument(
        '--recon-detection-dir',
        type=str,
        required=True,
        help='Directory containing reconstruction detection outputs (expects a preds/ subfolder).')
    parser.add_argument(
        '--gt-detection-dir',
        type=str,
        required=True,
        help='Directory containing ground-truth detection outputs (expects a preds/ subfolder).')
    parser.add_argument(
        '--recon-img-dir',
        type=str,
        required=True,
        help='Directory containing reconstructed images.')
    parser.add_argument(
        '--gt-img-dir',
        type=str,
        required=True,
        help='Directory containing ground-truth images.')
    parser.add_argument(
        '--intermediate-results-dir',
        type=str,
        default=None,
        help='Optional directory to save intermediate numpy outputs.')
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='W&B project name. If set, evaluation scores are logged to Weights & Biases.')
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='Optional W&B run name. ')

    return parser.parse_args()


def main():
    args = get_args()

    obj_f1 = eval_object_f1(args.recon_detection_dir, args.gt_detection_dir, verbose=True)
    effnet = eval_effnet(args.recon_img_dir, args.gt_img_dir)
    recon_captions, gt_captions, cap_sim = eval_cap_sim(args.recon_img_dir, args.gt_img_dir)

    if args.intermediate_results_dir is not None:
        os.makedirs(args.intermediate_results_dir, exist_ok=True)
        np.save(os.path.join(args.intermediate_results_dir, 'obj_f1.npy'), obj_f1)
        np.save(os.path.join(args.intermediate_results_dir, 'effnet.npy'), effnet)
        np.save(os.path.join(args.intermediate_results_dir, 'cap_sim.npy'), cap_sim)
        np.save(os.path.join(args.intermediate_results_dir, 'recon_captions.npy'), recon_captions)
        np.save(os.path.join(args.intermediate_results_dir, 'gt_captions.npy'), gt_captions)

    obj_f1_score = np.mean(obj_f1)
    effnet_score = np.mean(effnet)
    cap_sim_score = np.mean(cap_sim)
    
    print(f'Evaluation results for model: {args.model_name}')
    print(f'Object F1: {obj_f1_score:.3f}')
    print(f'EffNet: {effnet_score:.3f}')
    print(f'Caption Similarity: {cap_sim_score:.3f}')
    print(f'SEED: {(obj_f1_score + (1-effnet_score) + cap_sim_score) / 3:.3f}')

    if args.wandb_project is not None:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model_name': args.model_name,
            }
        )

        wandb.log({
            'eval/obj_f1': float(obj_f1_score),
            'eval/effnet': float(effnet_score),
            'eval/cap_sim': float(cap_sim_score),
            'eval/seed': float((obj_f1_score + (1-effnet_score) + cap_sim_score) / 3),
        })
        run.finish()



if __name__ == '__main__':
    main()
