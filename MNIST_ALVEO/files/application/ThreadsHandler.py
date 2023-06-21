import os
import threading
from time import sleep, time
import cv2
import numpy as np
from tensorboard import summary

from trt_pose import models
from Processing import preprocessingTransforms, postProcessBatch, \
    get_child_subgraph_dpu, runDPU, postProcessing, num_parts, num_links
import torch
from torchsummary import summary

extensionFramework = {
    ".xmodel": "vitisai",
    ".pth": "pytorch",
    ".onnx": "pytorch"
}

class RunningThread (threading.Thread):
    def __init__(self, handler, slot: int, inputBatch,
                 # framework: str,
                 dpuRunner = None, modelFile = None):
        threading.Thread.__init__(self)

        assert not((dpuRunner is None) and (modelFile is None))
        self.handler = handler
        self.slot = slot
        # self.framework = framework
        self.dpuRunner = dpuRunner
        self.modelFile = modelFile
        self.inputBatch = inputBatch

    def run(self):
        outputBatch = None

        # Pre-processing:
        inputBatch = []
        for img in self.inputBatch:
            inputBatch.append(preprocessingTransforms(img)[None, :])
        inputBatch = torch.cat(inputBatch)

        if self.dpuRunner != None:
            # DPU processing:
            middleBatch = runDPU(inputBatch, self.dpuRunner)
        else: # Pytorch
            # Processing with pytorch:
            # model = models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
            # model = models.resnet18_baseline_att(num_parts, 2 * num_links).eval()
            model = models.resnet18_baseline_att(num_parts, 2 * num_links).eval()
            # model = models.densenet121_baseline_att(num_parts, 2 * num_links).eval()
            MODEL_WEIGHTS = self.modelFile
            model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))

            # summary(model, inputBatch[0].size())

            middleBatch = model(inputBatch)

        # Post-processing:
        # outputBatch = postProcessing(*middleBatch)
        outputBatch = postProcessBatch(middleBatch, self.inputBatch)

        self.handler.writeOutputBatch(self.slot, outputBatch)

class DisplayThread (threading.Thread):
    def __init__(self, handler, targetFPS = 24):
        threading.Thread.__init__(self)
        self.handler = handler
        self.targetFPS = targetFPS
        self.lastTimestamp = round(time() * 1000)

    def display(self, batch):
        targetPeriod = (1.0 / float(self.targetFPS)) * 1000.0
        for k in range(0, batch.shape[0]):
            currentTimestamp = round(time() * 1000)
            diff = currentTimestamp - self.lastTimestamp
            waitTime = 0 if diff >= targetPeriod else targetPeriod - diff
            if waitTime > 0:
                sleep(waitTime / 1000)
            cv2.imshow("Output", batch[k])
            cv2.waitKey(1)
            print('Frame de salida ', k, ' (', int(waitTime), ')')
            self.lastTimestamp = round(time() * 1000)

    def run(self):
        while True:
            busySlot = self.handler.getFirstBusySlot()
            self.handler.outputWrittenEvents[busySlot].wait()
            self.handler.lock.acquire()
            batch = self.handler.outputBatchesQueue[busySlot]
            self.handler.lock.release()
            self.display(batch)

            self.handler.deallocateSlot()


class ThreadsHandler:
    def __init__(self, maxNumThreads: int, model):
        self.maxNumThreads = maxNumThreads
        self.firstBusySlot = 0
        self.firstEmptySlot = 0
        self.numRunningThreads = 0

        self.threadsQueue = [None] * maxNumThreads
        self.outputBatchesQueue = [None] * maxNumThreads


        self.outputWrittenEvents = [threading.Event()] * maxNumThreads
        self.queuesAreJustNotFullEvent = threading.Event()
        # self.runningThreadsSemaphore = threading.Semaphore(maxNumThreads)

        self.lock = threading.Lock()

        extension = os.path.splitext(model)[1]
        assert extension in extensionFramework
        self.framework = extensionFramework[extension]
        if extensionFramework[extension] == 'vitisai':
            g = xir.Graph.deserialize(model)
            subgraphs = get_child_subgraph_dpu(g)
            self.dpuRunners = []
            for i in range(maxNumThreads):
                self.dpuRunners.append(vart.Runner.create_runner(subgraphs[0], "run"))
        # elif extension == 'pytorch':
        else:
            # self.model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
            self.model = model


    def getFirstBusySlot(self):
        self.lock.acquire()
        res = self.firstBusySlot
        self.lock.release()
        return res

    def getFirstEmptySlot(self):
        self.lock.acquire()
        res = self.firstEmptySlot
        self.lock.release()
        return res

    def writeOutputBatch(self, slot, batch):
        # We reserve the handler:
        self.lock.acquire()
        # We write the batch:
        self.outputBatchesQueue[slot] = batch

        # We free the handler:
        self.lock.release()

        self.outputWrittenEvents[slot].set()

    def allocateSlot(self, inputBatch, waitForSpace=True):
        if waitForSpace:
            if self.areQueuesFull(blocking=True):
                self.queuesAreJustNotFullEvent.wait()

        self.lock.acquire()
        # self.runningThreadsSemaphore.acquire()
        if not waitForSpace:
            assert self.areQueuesFull(blocking=False)
        slot = self.firstEmptySlot
        if self.framework == 'vitisai':
            self.threadsQueue[slot] = RunningThread(self, slot, inputBatch, dpuRunner=self.dpuRunners[slot])
        else:
            self.threadsQueue[slot] = RunningThread(self, slot, inputBatch, modelFile=self.model)
        self.outputBatchesQueue[slot] = None
        self.numRunningThreads = self.numRunningThreads + 1
        if self.firstEmptySlot >= (self.maxNumThreads - 1):
            self.firstEmptySlot = 0
        else:
            self.firstEmptySlot = self.firstEmptySlot + 1
        self.threadsQueue[slot].start()
        self.lock.release()
        # self.outputWrittenLockQueue[slot].acquire()

        return slot

    def deallocateSlot(self):
        self.lock.acquire()
        assert not self.areQueuesEmpty(blocking=False)
        slot = self.firstBusySlot
        self.outputWrittenEvents[slot].clear()
        self.threadsQueue[slot].join()
        self.threadsQueue[slot] = None
        self.numRunningThreads = self.numRunningThreads - 1
        if self.firstBusySlot >= (self.maxNumThreads - 1):
            self.firstBusySlot = 0
        else:
            self.firstBusySlot = self.firstBusySlot + 1
        # self.runningThreadsSemaphore.release()

        if self.numRunningThreads == (self.maxNumThreads - 1):
            self.queuesAreJustNotFullEvent.set()

        self.lock.release()


        return slot


    def areQueuesEmpty(self, blocking=True):
        if blocking:
            self.lock.acquire()
        numRunningThreads = self.numRunningThreads
        if blocking:
            self.lock.release()
        return numRunningThreads <= 0 # and (self.firstEmptySlot == self.firstBusySlot)

    def areQueuesFull(self, blocking=True):
        if blocking:
            self.lock.acquire()
        numRunningThreads = self.numRunningThreads
        res = numRunningThreads >= self.maxNumThreads
        if res:
            self.queuesAreJustNotFullEvent.clear()
        if blocking:
            self.lock.release()
        return res # and (self.firstEmptySlot == self.firstBusySlot)


