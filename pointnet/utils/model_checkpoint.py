import torch
import os
import os.path as osp


class CheckpointManager:
    def __init__(
        self,
        dirpath: str,
        metric_name: str,
        mode: str = "min",
        topk: int = 1,
        verbose: bool = False,
    ):
        """
        dirpath: directory to save the model file.
        metric_name: the name of metric to track.
        mode: one of {min, max}. The decision to save current ckpt is based on
            either minimizing the quantity or maximizing the quantity.
            e.g., acc: max, loss: min
        topk: # of checkpoints to save.
        verbose: verbosity mode
        """
        self.dirpath = dirpath
        self.metric_name = metric_name
        self.mode = mode
        self.topk = topk
        self.verbose = verbose

        self._cache = []

        os.makedirs(self.dirpath, exist_ok=True)

    def update(self, model: torch.nn.Module, epoch: int, metric: float, fname: str):
        assert isinstance(epoch, int) and isinstance(metric, float)

        # filename = osp.join(self.dirpath, f"epoch={epoch}-{self.metric_name}={metric}.ckpt")
        filename = osp.join(self.dirpath, f"{fname}_epoch{epoch}_metric{metric}.ckpt")

        save_check = False
        if len(self._cache) < self.topk:
            save_check = True
        else:
            assert len(self._cache) <= self.topk

            for fn, met in self._cache:
                if self.mode == "min":
                    if metric < met:
                        save_check = True
                        break
                elif self.mode == "max":
                    if metric > met:
                        save_check = True
                        break

        if save_check:
            self._cache.append((filename, metric))
            assert not osp.exists(filename)
            torch.save(model.state_dict(), filename)
            if self.verbose:
                print(f"saving checkpoint to {filename}")

            # sort cache
            sorted_cache = sorted(
                self._cache, key=lambda x: x[1], reverse=self.mode == "max"
            )
            self._cache = sorted_cache[: self.topk]
            # delete an outdated checkpoint file.
            for fn, met in sorted_cache[self.topk :]:
                assert osp.exists(fn)
                os.system(f"rm {fn}")

    def load_best_ckpt(self, model, device):
        try:
            ckptname = self._cache[0][0]
            ckpt = torch.load(ckptname, map_location=device)
            model.load_state_dict(ckpt)
            print(f"loaded best ckpt from {ckptname}")
        except:
            print("cannot load checkpoint")
            

