import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import math
import time
import os
from thop import profile
from sggcmnet import SGGCMNet, DynamicWeightedLoss
from utils import (
    setup_seed, load_hsi_data, apply_pca, create_patches,
    split_data, AverageMeter, output_metric, DATASET_CLASS_NAMES
)
from generate_pic import get_cls_map


SOTA_SPLITS = {
    "IndianPines": (0.03, 0.97),
    "PaviaUniversity": (0.005, 0.995),
    "Houston": (0.02, 0.98),
}

parser = argparse.ArgumentParser("SGGCMNet Training for HSI Classification")
parser.add_argument('--gpu_id', default='0', help='GPU ID')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--dataset', choices=['IndianPines', 'PaviaUniversity', 'Houston'], default='IndianPines')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patch_size', type=int, default=11)
parser.add_argument('--pca_bands', type=int, default=30)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--runs', type=int, default=10)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


class HsiDataset(Data.Dataset):
    def __init__(self, patches, labels):
        self.patches = torch.from_numpy(patches).float()
        self.labels = torch.from_numpy(labels - 1).long()

    def __getitem__(self, index):
        return self.patches[index], self.labels[index]

    def __len__(self):
        return len(self.patches)


class ModelInferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, _, logits2 = self.model(x)
        logits = F.adaptive_avg_pool2d(logits2, 1).squeeze(-1).squeeze(-1)
        return logits, None


def tcmd_loss(p, q, temperature=2.0):
    """Temperature-Regulated Cross-Stage Mutual Distillation: symmetric KL with temperature scaling."""
    T = temperature
    kl_pq = F.kl_div(
        F.log_softmax(p / T, dim=1),
        F.softmax(q / T, dim=1).detach(),
        reduction='batchmean'
    ) * (T ** 2)
    kl_qp = F.kl_div(
        F.log_softmax(q / T, dim=1),
        F.softmax(p / T, dim=1).detach(),
        reduction='batchmean'
    ) * (T ** 2)
    return 0.5 * (kl_pq + kl_qp)


def main():
    all_oa, all_aa, all_kappa = [], [], []
    all_test_times = []
    all_aa_per_class = []
    all_matrices = []

    results_dir = f"results/{args.dataset}_results"
    os.makedirs(results_dir, exist_ok=True)

    for i in range(args.runs):
        print(f"--- Run {i+1}/{args.runs} ---")

        run_seed = args.seed + i
        setup_seed(run_seed)

        hsi_data, gt_labels = load_hsi_data(args.dataset)
        num_classes = np.max(gt_labels)
        hsi_pca = apply_pca(hsi_data, args.pca_bands)

        train_ratio, test_ratio = SOTA_SPLITS.get(args.dataset, (0.1, 0.9))
        patches, labels = create_patches(hsi_pca, gt_labels, args.patch_size)
        X_train, X_test, y_train, y_test = split_data(patches, labels, train_ratio, test_ratio, run_seed)

        X_train = X_train.transpose(0, 3, 1, 2)
        X_test = X_test.transpose(0, 3, 1, 2)

        train_dataset = HsiDataset(X_train, y_train)
        test_dataset = HsiDataset(X_test, y_test)
        train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        # 4 loss terms: 3 CE (progressive deep supervision) + 1 TCMD (mutual distillation)
        model = SGGCMNet(
            in_channels=args.pca_bands,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
        ).cuda()

        criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        dynamic_loss_fn = DynamicWeightedLoss(num_losses=4).cuda()

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(dynamic_loss_fn.parameters()),
            lr=args.lr, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        if i == 0:
            model.eval()
            dummy = torch.randn(1, args.pca_bands, args.patch_size, args.patch_size).cuda()
            flops, params = profile(model, inputs=(dummy,))
            model.train()

        print(f"  - Parameters: {params / 1e6:.4f}M")
        print(f"  - FLOPs: {flops / 1e9:.4f}G")

        for epoch in range(args.epochs):
            model.train()
            train_loss_avg = AverageMeter()

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                optimizer.zero_grad()

                logits0, logits1, logits2 = model(batch_x)

                cls0 = F.adaptive_avg_pool2d(logits0, 1).squeeze(-1).squeeze(-1)
                cls1 = F.adaptive_avg_pool2d(logits1, 1).squeeze(-1).squeeze(-1)
                cls2 = F.adaptive_avg_pool2d(logits2, 1).squeeze(-1).squeeze(-1)

                # Progressive deep supervision: CE loss per stage
                loss_ce0 = criterion_cls(cls0, batch_y)
                loss_ce1 = criterion_cls(cls1, batch_y)
                loss_ce2 = criterion_cls(cls2, batch_y)

                # TCMD: temperature-regulated mutual distillation across stages
                loss_mutual = 0.5 * (
                    tcmd_loss(cls0, cls1, temperature=2.0) +
                    tcmd_loss(cls1, cls2, temperature=2.0) +
                    tcmd_loss(cls0, cls2, temperature=2.0)
                ) / 3.0

                loss = dynamic_loss_fn([loss_ce0, loss_ce1, loss_ce2, loss_mutual])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_avg.update(loss.item(), batch_x.size(0))

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
                w = dynamic_loss_fn.get_weights()
                lr_now = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {train_loss_avg.avg:.4f} | lr: {lr_now:.1e} | W: [s0:{w[0]:.2f}, s1:{w[1]:.2f}, s2:{w[2]:.2f}, mu:{w[3]:.2f}]")

        # Evaluate on test set (main output from cls_head2)
        tic = time.time()
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, _, logits2 = model(batch_x)
                logits = F.adaptive_avg_pool2d(logits2, 1).squeeze(-1).squeeze(-1)
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        toc = time.time()
        test_time = toc - tic

        oa, kappa, aa, aa_per_class, matrix = output_metric(np.array(all_targets), np.array(all_preds))
        print(f"OA: {oa:.4f}, Kappa: {kappa:.4f}, AA: {aa:.4f}")

        run_stats_filename = os.path.join(results_dir, f"run_{i+1}_stats.txt")
        with open(run_stats_filename, 'w') as f:
            f.write(f"--- Performance Stats for {args.dataset} (Run {i+1}) ---\n")
            f.write(f"Seed: {run_seed}\n")
            f.write(f"Parameters (M): {params / 1e6:.4f}\n")
            f.write(f"FLOPs (G): {flops / 1e9:.4f}\n")
            f.write(f"Testing Time (seconds): {test_time:.4f}\n\n")
            f.write(f"Overall Accuracy (OA): {oa:.4f}\n")
            f.write(f"Average Accuracy (AA): {aa:.4f}\n")
            f.write(f"Cohen's Kappa (Kappa): {kappa:.4f}\n\n")
            class_names = DATASET_CLASS_NAMES.get(args.dataset, [])
            for j, class_acc in enumerate(aa_per_class):
                class_name = class_names[j] if j < len(class_names) else f"Class {j+1}"
                f.write(f"{class_name}: {class_acc:.4f}\n")
            f.write("\n--- Confusion Matrix ---\n")
            f.write(np.array2string(matrix))

        all_oa.append(oa)
        all_aa.append(aa)
        all_kappa.append(kappa)
        all_test_times.append(test_time)
        all_aa_per_class.append(aa_per_class)
        all_matrices.append(matrix)

        if i == args.runs - 1:
            all_patches, all_labels = create_patches(hsi_pca, gt_labels, args.patch_size)
            all_patches = all_patches.transpose(0, 3, 1, 2)
            all_dataset = HsiDataset(all_patches, all_labels)
            all_loader = Data.DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            save_path = f'results/classification_maps/Run_{i+1}/'
            os.makedirs(save_path, exist_ok=True)
            wrapped_model = ModelInferenceWrapper(model)
            get_cls_map(wrapped_model, all_loader, gt_labels, save_path, args.dataset)

    class_names = DATASET_CLASS_NAMES.get(args.dataset, [])
    if args.runs > 1:
        print("\n--- Averaged Results ---")
        print(f"OA: {np.mean(all_oa):.4f} ± {np.std(all_oa):.4f}")
        print(f"AA: {np.mean(all_aa):.4f} ± {np.std(all_aa):.4f}")
        print(f"Kappa: {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")

        all_aa_per_class = np.array(all_aa_per_class)
        mean_per_class = np.mean(all_aa_per_class, axis=0)
        std_per_class = np.std(all_aa_per_class, axis=0)
        mean_matrix = np.mean(all_matrices, axis=0)

        stats_filename = f"results/{args.dataset}_avg_stats.txt"
        with open(stats_filename, 'w') as f:
            f.write(f"--- Averaged Performance Stats for {args.dataset} ({args.runs} runs) ---\n")
            f.write(f"Parameters (M): {params / 1e6:.4f}\n")
            f.write(f"FLOPs (G): {flops / 1e9:.4f}\n\n")
            f.write(f"Overall Accuracy (OA): {np.mean(all_oa):.4f} ± {np.std(all_oa):.4f}\n")
            f.write(f"Average Accuracy (AA): {np.mean(all_aa):.4f} ± {np.std(all_aa):.4f}\n")
            f.write(f"Cohen's Kappa (Kappa): {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}\n\n")
            f.write("--- Per-Class Accuracy ---\n")
            for idx, (mean_acc, std_acc) in enumerate(zip(mean_per_class, std_per_class)):
                class_name = class_names[idx] if idx < len(class_names) else f"Class {idx+1}"
                f.write(f"{class_name}: {mean_acc:.4f} ± {std_acc:.4f}\n")
            f.write("\n--- Averaged Confusion Matrix ---\n")
            f.write(np.array2string(mean_matrix, formatter={'float_kind':lambda x: "%.2f" % x}))

        print(f"Averaged stats saved to {stats_filename}")


if __name__ == '__main__':
    main()
