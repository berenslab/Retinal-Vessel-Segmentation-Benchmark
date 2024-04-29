import time
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import double_threshold_iteration
from utils.metrics import get_metrics, get_metrics, count_connect_component
import ttach as tta


class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint, test_loader, device):
        super(Trainer, self).__init__()
        self.device = device
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader

        self.model = model.to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if device == 'cuda':
            cudnn.benchmark = True

    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, total=len(self.test_loader))
        tic = time.time()

        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.to(self.device)   #.cuda(non_blocking=True)
                gt = gt.to(self.device)     #.cuda(non_blocking=True)
                pre = self.model(img)
                if isinstance(pre, tuple):
                    logits_aux, logits = pre
                    loss_aux = self.loss(logits_aux, gt)
                    loss = loss_aux + self.loss(logits, gt)
                    pre = pre[1]
                else:

                    loss = self.loss(pre, gt)

                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)

                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(pre, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC: # use count connect component
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(
                        *get_metrics(pre, gt, self.CFG.threshold).values())
                    if self.CFG.CCC: 
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} MCC {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')

        
        