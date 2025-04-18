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
        print("prec")
        print("tp:", tp)
        print("fp:", fp)
        print("end")

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
        window_size = 1
        for idx in range(len(ds)):
            start = max(0, idx - window_size)
            end = min(len(ds), idx + window_size)
            local_recon_losses = recon_losses[start:end]
            adaptive_threshold = 0.9 * torch.mean(local_recon_losses) + 0.5 * torch.std(local_recon_losses)
            adaptive_thresholds.append(adaptive_threshold)

        tp = 0
        fn = 0
        tp_1000=0
        tp_2000=0
        for idx in range(len(ds)):
            if ds[idx][3].item() in params["global_anomaly_dept"] or ds[idx][3].item() in params["local_anomaly_dept"]:
                if recon_losses[idx] > adaptive_thresholds[idx]:
                    tp += 1
                    if(ds[idx][3].item()==1000):
                      tp_1000+=1
                    else: 
                      tp_2000+=1
                else:
                    fn += 1

        rec = tp / float(tp + fn)

        info = {
            "rec_losses": recon_losses.detach().cpu().numpy().tolist(),
            "depts": ds[:][3].cpu().numpy().tolist()
        }
        # with open("rec_loss_log.txt", "w") as f:
        #   for idx in range(len(ds)):
        #       dept = ds[idx][3].item()
        #       loss_val = recon_losses[idx].item()
        #       threshold_val = adaptive_thresholds[idx].item()
        #       f.write(f"{idx}, dept={dept}, loss={loss_val:.6f}, threshold={threshold_val:.6f}\n")
        print("rec")
        print("tp:", tp)
        print("fp:", fn)
        print("tp1000:", tp_1000)
        print("tp2000:", tp_2000)
        print("end")

        return rec, info