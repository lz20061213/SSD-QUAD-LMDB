# --------------------------------------------------------
# SSD @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import os

import dragon.core.mpi as mpi
import dragon.vm.caffe as caffe
import google.protobuf as pb2
from dragon.config import logger
from dragon.vm.caffe.proto import caffe_pb2

from config import cfg
from core.utils.timer import Timer


class SolverWrapper(object):
    def __init__(self, solver_prototxt, output_dir,
                 pretrained_model=None):

        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)

        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def train_model(self, max_iters, warm_up):
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        # lower the learning rate then warm-up
        if warm_up != 0:
            self.solver._optimizer.lr *= self.solver_param.gamma
        while self.solver.iter < max_iters:
            if warm_up > 0 and self.solver.iter == warm_up:
                self.solver._optimizer.lr /= self.solver_param.gamma
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                logger.info('speed: {:.3f}s / iter'.format(timer.average_time))

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

    def snapshot(self):
        if mpi.Is_Init():
            if not mpi.AllowSnapshot(): return
        net = self.solver.net
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)
        net.save(str(filename))
        logger.info('Wrote snapshot to: {:s}'.format(filename))
        return filename

    def restore(self, start_itet, snapshot_model):
        print ('Restoring snapshot model '
               'weights from {:s}').format(snapshot_model)

        net = self.solver.net
        self.solver._iter = start_itet
        net.copy_from(snapshot_model)

def train_net(solver_txt, output_dir,
              pretrained_model=None, snapshot_model=None,
              start_iter=0, max_iters=60000, warm_up=0):

    sw = SolverWrapper(solver_txt, output_dir,
                       pretrained_model=pretrained_model)

    if snapshot_model is not None:
        sw.restore(start_iter, snapshot_model)

    logger.info('Solving...')
    model_paths = sw.train_model(max_iters, warm_up)
    logger.info('done solving')
    return model_paths