import os
import json
import csv
import time
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
import numpy as np
import torch
import matplotlib.pyplot as plt

def make_run_dir(run_base_dir: str, run_name: str) -> str:
    os.makedirs(run_base_dir, exist_ok=True)
    run_dir = os.path.join(run_base_dir, run_name)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    i = 1
    while True:
        candidate = f"{run_dir}_{i:03d}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        i += 1


def setup_logger(log_path: str, logger_name: str = "train", also_console: bool = True,
                 mode: str = "w") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (notebooks / repeated runs)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if also_console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.propagate = False
    return logger


def write_config_json(config_path: str, config: dict) -> None:
    def _default(o):
        return str(o)  # last-resort for non-JSON types
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=_default)


class TrainingTracker:
    def __init__(
        self,
        run_base_dir: str,
        run_name: str,
        enable_logger: bool = True,
        enable_config: bool = True,
        enable_metrics: bool = True,
        also_console: bool = True,
        log_every: int = 10,
        monitor: str = "pi_val_loss",
        mode: str = "min",  # "min" for loss, "max" for score/accuracy
        # A SECOND metric to keep a best checkpoint for, saved as checkpoints/best_<monitor2>.pt.
        # Early stopping and best.pt still follow `monitor` alone.
        monitor2: Optional[str] = None,
        mode2: str = "min",
        config: Optional[Dict[str, Any]] = None,
        save_last: bool = True,
        save_full_model_object: bool = False,
        enable_plot: bool = True,
        plot_keys: tuple = ("train_loss", "val_loss",),  ### Just for plotting
        ## For .csv it automatically saves the data provided 
        patience: int = 20,
        # Continue an interrupted run in its own folder (run_base_dir/run_name must exist): the log is
        # appended to, metrics.csv and config.json are kept, and restore() reloads the counters.
        resume: bool = False,
    ) -> None:
        # user params
        self.run_base_dir = run_base_dir
        self.run_name = run_name
        self.resume = resume

        self.enable_logger = enable_logger
        self.enable_config = enable_config
        self.enable_metrics = enable_metrics
        self.also_console = also_console

        if log_every <= 0:
            raise ValueError("log_every must be >= 1")
        self.log_every = log_every

        self.monitor = monitor
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

        self.monitor2 = monitor2
        if mode2 not in ("min", "max"):
            raise ValueError("mode2 must be 'min' or 'max'")
        self.mode2 = mode2

        self.config = config or {}

        self.save_last = save_last
        self.save_full_model_object = save_full_model_object

        self.enable_plot = enable_plot
        self.plot_keys = plot_keys

        # internal state placeholders
        self.run_dir: Path = None  # type: ignore
        self.ckpt_dir: Path = None  # type: ignore
        self.best_path: Path = None  # type: ignore
        self.last_path: Path = None  # type: ignore
        self.metrics_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.logger: Optional[logging.Logger] = None

        self.best_value: float = float("inf") if self.mode == "min" else -float("inf")
        self.best_epoch: int = -1
        self.best2_path: Path = None  # type: ignore
        self.best2_value: float = float("inf") if self.mode2 == "min" else -float("inf")
        self.best2_epoch: int = -1
        self.patience: int = patience

        self._last_improve_epoch: int = -1

        self._csv_header_written: bool = False
        self._csv_header: Optional[Iterable[str]] = None

        self._start_time: float = 0.0
        self._start_date: str = ""
        self._last_epoch: int = -1

        # init filesystem + logger/config/metrics
        self._init_run()

    def _init_run(self) -> None:
        self._start_time = time.monotonic()
        self._start_date = datetime.datetime.now().isoformat(timespec="seconds")

        # --- dirs
        if self.resume:
            self.run_dir = Path(self.run_base_dir) / self.run_name
            if not (self.run_dir / "checkpoints").is_dir():
                raise FileNotFoundError(f"resume: {self.run_dir} has no checkpoints/")
        else:
            self.run_dir = Path(make_run_dir(self.run_base_dir, self.run_name))
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_path = self.ckpt_dir / "best.pt"
        if self.monitor2:
            self.best2_path = self.ckpt_dir / f"best_{self.monitor2}.pt"
        self.last_path = self.ckpt_dir / "last.pt"

        # --- metrics
        if self.enable_metrics:
            self.metrics_path = self.run_dir / "metrics.csv"
            if not self.resume:
                with open(self.metrics_path, "w", newline="") as f:
                    pass

        # --- config (on resume the original one is kept: it already describes the run)
        if self.enable_config:
            self.config_path = self.run_dir / "config.json"
            if not self.resume:
                write_config_json(str(self.config_path), self.config)

        # --- logger
        if self.enable_logger:
            self.log_path = self.run_dir / "training.log"
            self.logger = setup_logger(
                str(self.log_path),
                logger_name=f"train_{self.run_dir.name}",
                also_console=self.also_console,
                mode="a" if self.resume else "w",
            )
            self.logger.info("Training resumed (new process)." if self.resume else "Training started.")
            self.logger.info(f"run_dir: {self.run_dir}")
            self.logger.info(f"log_every: {self.log_every}")

    def resume_checkpoint(self) -> Dict[str, Any]:
        """The checkpoint to continue from: the most recent of last.pt (every log_every epochs) and
        best.pt (every improvement). Taking the later one keeps best.pt consistent with the run:
        an improvement saved after the last last.pt is never left behind as a stale best.pt."""
        cks = []
        for p in (self.last_path, self.best_path):
            if p.exists():
                try:
                    cks.append(torch.load(str(p), map_location="cpu", weights_only=False))
                except Exception as exc:          # e.g. truncated by a crash during the write
                    if self.logger:
                        self.logger.warning(f"resume: {p.name} unreadable ({exc}) - not used")
        if not cks:
            raise FileNotFoundError(f"resume: no last.pt / best.pt in {self.ckpt_dir}")
        return max(cks, key=lambda c: int(c["epoch"]))

    def restore(self, ck: Dict[str, Any]) -> int:
        """Reload the counters from checkpoint `ck` (see resume_checkpoint) and return its epoch:
        best value/epoch, the second monitor, metrics.csv cut back to that epoch (the rows after it
        are recomputed), and the training clock (time already spent up to that epoch; the epochs lost
        to the interruption and the restart itself are not counted)."""
        epoch = int(ck["epoch"])
        self.best_value, self.best_epoch, self._last_epoch = float(ck["best_value"]), int(ck["best_epoch"]), epoch
        if self.monitor2 and self.best2_path.exists():
            b2 = torch.load(str(self.best2_path), map_location="cpu", weights_only=False)
            if int(b2["epoch"]) <= epoch:
                self.best2_value = float(b2["extra"]["metrics"][self.monitor2])
                self.best2_epoch = int(b2["epoch"])
            elif self.logger:
                self.logger.warning(f"resume: {self.best2_path.name} is from epoch {b2['epoch']} > {epoch}; "
                                    f"it is kept until the resumed run improves on it")
        lost = None
        if self.enable_metrics and self.metrics_path.exists():
            with open(self.metrics_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                self._csv_header = list(reader.fieldnames or [])
                rows = list(reader)
            keep = [r for r in rows if int(r["epoch"]) <= epoch]
            lost = len(rows) - len(keep)
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_header)
                writer.writeheader()
                writer.writerows(keep)
            self._csv_header_written = True
        stamps = self._log_stamps(epoch)
        if stamps[0] is not None:
            self._start_date = stamps[0].isoformat(timespec="seconds")
        elapsed = ck.get("elapsed_s")
        if elapsed is None and None not in stamps:        # checkpoints written before elapsed_s was saved
            elapsed = (stamps[1] - stamps[0]).total_seconds()
        self._start_time = time.monotonic() - float(elapsed or 0.0)
        if self.logger:
            self.logger.info(f"Resumed from epoch {epoch} ({lost} later metrics rows dropped, recomputed); "
                             f"best_{self.monitor}={self.best_value:.6g} at epoch={self.best_epoch}; "
                             f"time already spent {elapsed or 0:.0f} s")
        return epoch

    def _log_stamps(self, epoch: int):
        """(start of the first segment, end of `epoch`) from training.log timestamps, or None."""
        start = end = None
        if self.log_path is None or not self.log_path.exists():
            return start, end
        tags = (f"| epoch={epoch} |", f"at epoch={epoch} ->")
        with open(self.log_path, "r", errors="replace") as f:
            for line in f:
                try:
                    t = datetime.datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
                except ValueError:
                    continue
                if start is None and "Training started." in line:
                    start = t
                if end is None and any(s in line for s in tags):
                    end = t
        return start, end

    def _is_improvement(self, value: float) -> bool:
        return (value < self.best_value) if self.mode == "min" else (value > self.best_value)

    def _write_metrics_row(self, row_dict: Dict[str, Any]) -> None:
        if not self.enable_metrics or self.metrics_path is None:
            return

        if not self._csv_header_written:
            self._csv_header = list(row_dict.keys())
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_header)
                writer.writeheader()
                writer.writerow(row_dict)
            self._csv_header_written = True
            return

        missing = [k for k in self._csv_header if k not in row_dict]
        extra = [k for k in row_dict if k not in self._csv_header]
        if missing or extra:
            raise ValueError(
                f"Metrics keys changed.\nMissing: {missing}\nExtra: {extra}\n"
                f"Expected header: {self._csv_header}"
            )

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_header)
            writer.writerow(row_dict)

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra = extra or {}
        if self.save_full_model_object:
            payload = {
                "epoch": epoch,
                "model": model,
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "extra": extra,
                "monitor": self.monitor,
                "mode": self.mode,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch,
            }
        else:
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "extra": extra,
                "monitor": self.monitor,
                "mode": self.mode,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch,
            }
        # for a resume: the training time so far, and the global RNG states at the end of this epoch,
        # against which the resumed process checks its replay (numpy's as plain Python values)
        name, keys, pos, has_gauss, cached = np.random.get_state()
        payload["elapsed_s"] = time.monotonic() - self._start_time
        payload["rng"] = {"torch": torch.get_rng_state(),
                          "numpy": [name, keys.tolist(), int(pos), int(has_gauss), float(cached)]}
        # write-then-rename: a write that fails half-way (e.g. a full disk) leaves the previous
        # checkpoint intact instead of a truncated file a resume could not load
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, str(tmp))
        os.replace(tmp, path)

    def plot_loss(self) -> None:
        if not self.enable_plot or not self.enable_metrics or self.metrics_path is None:
            return

        png_path = str(self.run_dir / "loss_history.png")

        # Ensure plot_keys is iterable and non-empty
        plot_keys = list(self.plot_keys) if self.plot_keys is not None else []
        if not plot_keys:
            return

        epochs: List[int] = []
        series: Dict[str, List[float]] = {k: [] for k in plot_keys}

        with open(self.metrics_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "epoch" not in fieldnames:
                return

            # Keep only keys that exist in the CSV
            valid_keys = [k for k in plot_keys if k in fieldnames]
            if not valid_keys:
                return

            # Rebuild series dict to contain only valid keys
            series = {k: [] for k in valid_keys}

            for row in reader:
                # Skip rows with missing epoch
                if not row.get("epoch"):
                    continue

                try:
                    ep = int(row["epoch"])
                except ValueError:
                    continue

                # Parse available values; if any key missing/invalid in this row, store NaN
                epochs.append(ep)
                for k in valid_keys:
                    val = row.get(k, "")
                    try:
                        series[k].append(float(val))
                    except (ValueError, TypeError):
                        series[k].append(float("nan"))

        if not epochs:
            return

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.set_yscale("log")

        # Labels: nicer defaults for common names, otherwise use the key
        pretty = {
            "train_loss": "Training loss",
            "val_loss": "Validation loss",
            "loss_train": "Training loss",
            "loss_val": "Validation loss",
            "training_loss": "Training loss",
            "validation_loss": "Validation loss",
        }

        for k, y in series.items():
            ax.plot(epochs, y, label=pretty.get(k, k))

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss history")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path, format="png")
        plt.close(fig)

    def early_stop_check(self, *, epoch: int) -> bool:
        """
        Returns True if training should stop early.
        """

        # Stop if no improvement for `patience` epochs
        if (epoch - self.best_epoch) >= self.patience:
            if self.logger:
                self.logger.info(
                    f"Early stopping triggered at epoch={epoch}. "
                    f"No improvement in {self.patience} epochs. "
                    f"best_{self.monitor}={self.best_value:.6g} at epoch={self.best_epoch}."
                )
            return True

        return False

    def update(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **metrics: float,
    ) -> Dict[str, Any]:
        
## provide the metrics in the following way trainer.update(epoch, ..., train_loss = xxx, val_loss = yyy, accuracy_loss = zzz, ...)
## the code creates itself a dictionary metrics = {'train_loss':xxx, 'val_loss':yyy, ... }

        if self.monitor not in metrics:
            raise ValueError(f"monitor='{self.monitor}' not found in metrics keys: {list(metrics.keys())}")

        monitor_value = float(metrics[self.monitor])
        self._last_epoch = epoch

        # always store metrics row
        self._write_metrics_row({"epoch": epoch, **metrics})

        # best checkpoint: checked EVERY epoch (it used to sit inside the log_every block, so only
        # epochs that were multiples of log_every could become best.pt - the true optimum was lost)
        improved = self._is_improvement(monitor_value)
        if improved:
            self.best_value = monitor_value
            self.best_epoch = epoch
            self._save_checkpoint(self.best_path, epoch, model, optimizer, extra={"metrics": metrics})
            if self.logger:
                self.logger.info(f"New best: {self.monitor}={monitor_value:.2e} at epoch={epoch} -> {self.best_path}")

        # second best checkpoint - tracked, never used for early stopping
        if self.monitor2 and self.monitor2 in metrics:
            v2 = float(metrics[self.monitor2])
            if (v2 < self.best2_value) if self.mode2 == "min" else (v2 > self.best2_value):
                self.best2_value, self.best2_epoch = v2, epoch
                self._save_checkpoint(self.best2_path, epoch, model, optimizer,
                                      extra={"metrics": metrics})
                if self.logger:
                    self.logger.info(f"New best2: {self.monitor2}={v2:.2e} at epoch={epoch} "
                                     f"-> {self.best2_path}")

        if epoch % self.log_every == 0:

            if self.logger:
                msg = " | ".join([f"epoch={epoch}"] + [f"{k}={metrics[k]:.2e}" for k in metrics])
                self.logger.info(msg)

            if self.save_last:
                self._save_checkpoint(self.last_path, epoch, model, optimizer, extra={"metrics": metrics})

            self.plot_loss()

        # return {
        #     "improved": improved,
        #     "best_value": self.best_value,
        #     "best_epoch": self.best_epoch,
        #     "run_dir": str(self.run_dir),
        #     "best_path": str(self.best_path),
        #     "last_path": str(self.last_path)
        # }

    def close(self) -> None:
        elapsed_s = time.monotonic() - self._start_time
        h, rem = divmod(int(elapsed_s), 3600)
        m, s   = divmod(rem, 60)

        summary = {
            "date_started":         self._start_date,
            "date_finished":        datetime.datetime.now().isoformat(timespec="seconds"),
            "training_time_s":      round(elapsed_s, 1),
            "training_time_hms":    f"{h:02d}:{m:02d}:{s:02d}",
            "actual_epochs_trained": self._last_epoch + 1,
            "best_epoch":           self.best_epoch,
            f"best_{self.monitor}": self.best_value,
        }

        if self.enable_config and self.config_path is not None and self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    saved = json.load(f)
                saved["run_summary"] = summary
                write_config_json(str(self.config_path), saved)
            except Exception:
                pass  # never crash on bookkeeping

        if self.logger:
            self.logger.info(
                f"Training finished | time={summary['training_time_hms']} | "
                f"best_{self.monitor}={self.best_value:.4e} @ epoch {self.best_epoch}"
            )
            for h in list(self.logger.handlers):
                h.close()
                self.logger.removeHandler(h)
            self.logger = None

        # Export portable state-dicts so load_model can use weights_only=True
        # (avoids pickle class-path issues when loading from a different project).
        try:
            from .utils import export_state_dict
            for _ckpt in ("best.pt", "last.pt"):
                if (self.ckpt_dir / _ckpt).exists():
                    export_state_dict(self.run_dir, _ckpt)
        except Exception as _exc:
            print(f"[TrainingTracker] state-dict export skipped: {_exc}")
