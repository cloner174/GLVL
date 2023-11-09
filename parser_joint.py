import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")  # default = 4  \12 out memory
    parser.add_argument("--infer_batch_size", type=int, default=4,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=1000,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=300)  # default = 3
    parser.add_argument("--lr", type=float, default=0.0001, help="_")  # default = 0.00001
    parser.add_argument("--lr_crn_layer", type=float, default=5e-3, help="Learning rate for the CRN layer")
    parser.add_argument("--lr_crn_net", type=float, default=5e-4,
                        help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")  # default = 1000
    parser.add_argument("--queries_per_epoch", type=int, default=3000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")  # default = 5000
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")  # default = 10
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")  # default = 1000
    parser.add_argument("--mining", type=str, default="full", choices=["partial", "full", "random", "msls_weighted"])
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet50conv5",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default='gem',
                        help='netvlad,  gem,  spoc,  mac, rmac, crn, rrm, cls, seqpool')
    parser.add_argument('--netvlad_clusters', type=int, default=64,
                        help="Number of clusters for NetVLAD layer.")  # default = 64
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")  # 2048
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")  # imagenet
    parser.add_argument("--off_the_shelf", type=str, default="imagenet",
                        choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=32, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[500, 500], nargs=2,
                        help="Resizing shape for images (HxW).")  # default 480 640
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop",
                                 "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01,
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=0, help="_")
    parser.add_argument("--contrast", type=float, default=0, help="_")
    parser.add_argument("--saturation", type=float, default=0, help="_")
    parser.add_argument("--hue", type=float, default=0, help="_")
    parser.add_argument("--rand_perspective", type=float, default=0, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0, help="_")
    parser.add_argument("--random_rotation", type=float, default=0, help="_")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default='/home/lhl/data/data/dataset/', help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="Manchester",
                        help="Relative path of the dataset")  # pitts30k \ ICRA2022 \TianJin \village\ AID \Manchester
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--config", type=str, default='/home/lhl/data/data/visgeoloca/superpoint_RSSDIVCS_train.yaml',
                        help="Configs of SuperPointNet training")


    ### eval_joint
    parser.add_argument("--test_datasets_folder", type=str, default='/home/lhl/data/data/dataset/village',
                        help="Path with all datasets")

    parser.add_argument("--test_dataset_name", type=str, default="village",
                        help="Relative path of the dataset")


    args = parser.parse_args()

    if args.datasets_folder is None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")

    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")

    return args
