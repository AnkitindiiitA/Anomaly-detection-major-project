import math
import torch
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset


class MetricHandler(object):

    def _init_(self):
        pass

    def compute_prec_ratio(self, strategy, ds, params):
        """
        Computes Precision using adaptive thresholding.
        """
        b_size = 32
        n_batches = math.ceil(len(ds) / b_size)

        criterion = torch.nn.BCELoss(reduction="none")
        recon_losses = []
        for i in range(n_batches):
            start = i * b_size
            end = (i + 1) * b_size
            x = ds[start:end][0].to(strategy.device)
            pred = strategy.model(x)
            loss = criterion(pred, x)
            loss = torch.mean(loss, dim=1)
            recon_losses.append(loss)
        recon_losses = torch.cat(recon_losses, dim=0)

        adaptive_thresholds = []
        window_size = 10
        for idx in range(len(ds)):
            start = max(0, idx - window_size)
            end = min(len(ds), idx + window_size)
            local_recon_losses = recon_losses[start:end]
            adaptive_threshold = 0.9 * torch.mean(local_recon_losses) + 2.8 * torch.std(local_recon_losses)
            adaptive_thresholds.append(adaptive_threshold)

        tp = 0
        fp = 0
        for idx in range(len(ds)):
            if recon_losses[idx] > adaptive_thresholds[idx]:
                if ds[idx][3].item() in params["global_anomaly_dept"] or ds[idx][3].item() in params["local_anomaly_dept"]:
                    tp += 1
                else:
                    fp += 1

        prec = tp / float(tp + fp)

        info = {
            "rec_losses": recon_losses.detach().cpu().numpy().tolist(),
            "depts": ds[:][3].cpu().numpy().tolist()
        }

        return prec, info

    def compute_rec_ratio(self, strategy, ds, params):
        """
        Computes Recall using adaptive thresholding.
        """
        b_size = 32
        n_batches = math.ceil(len(ds) / b_size)

        criterion = torch.nn.BCELoss(reduction="none")
        recon_losses = []
        for i in range(n_batches):
            start = i * b_size
            end = (i + 1) * b_size
            x = ds[start:end][0].to(strategy.device)
            pred = strategy.model(x)
            loss = criterion(pred, x)
            loss = torch.mean(loss, dim=1)
            recon_losses.append(loss)
        recon_losses = torch.cat(recon_losses, dim=0)

        adaptive_thresholds = []
        window_size = 10
        for idx in range(len(ds)):
            start = max(0, idx - window_size)
            end = min(len(ds), idx + window_size)
            local_recon_losses = recon_losses[start:end]
            adaptive_threshold = 0.9 * torch.mean(local_recon_losses) + 2.8 * torch.std(local_recon_losses)
            adaptive_thresholds.append(adaptive_threshold)

        tp = 0
        fn = 0
        for idx in range(len(ds)):
            if ds[idx][3].item() in params["global_anomaly_dept"] or ds[idx][3].item() in params["local_anomaly_dept"]:
                if recon_losses[idx] > adaptive_thresholds[idx]:
                    tp += 1
                else:
                    fn += 1

        rec = tp / float(tp + fn)

        info = {
            "rec_losses": recon_losses.detach().cpu().numpy().tolist(),
            "depts": ds[:][3].cpu().numpy().tolist()
        }

        return rec, info