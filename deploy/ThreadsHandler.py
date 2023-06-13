import threading
from time import sleep, time
import cv2
import torchvision
from PIL import Image
import numpy as np

import vart
import xir

from Processing import preprocessingTransforms, get_child_subgraph_dpu, runDPU, postProcessing

class RunningThread (threading.Thread):
    def __init__(self, handler, slot: int, inputBatch, dpuRunner):
        threading.Thread.__init__(self)
        self.handler = handler
        self.slot = slot
        self.dpuRunner = dpuRunner
        self.inputBatch = inputBatch

    def run(self):
        outputBatch = None

        # Pre-processing:
        inputBatch = preprocessingTransforms(self.inputBatch)

        # DPU processing:
        middleBatch = runDPU(inputBatch, self.dpuRunner)

        # Post-processing:
        outputBatch = postProcessing(middleBatch, self.inputBatch)

        self.handler.writeOutputBatch(self.slot, outputBatch)

class DisplayThread (threading.Thread):
    def __init__(self, handler, targetFPS = 24):
        threading.Thread.__init__(self)
        self.handler = handler
        self.targetFPS = targetFPS
        self.lastTimestamp = round(time.time() * 1000)

    def display(self, batch):
        targetPeriod = (1.0 / float(self.targetFPS)) * 1000.0
        for k in range(0, batch.shape[0]):
            currentTimestamp = round(time.time() * 1000)
            diff = currentTimestamp - self.lastTimestamp

            waitTime = 0 if diff >= targetPeriod else targetPeriod - diff
            if waitTime > 0:
                sleep(waitTime)

            cv2.imshow("Output", batch[k])
            self.lastTimestamp = round(time.time() * 1000)

    def run(self):
        while True:
            busySlot = self.handler.getFirstBusySlot()
            self.handler.outputWrittenEvents[busySlot].wait()

            batch = self.handler.outputBatchesQueue[busySlot]
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

        g = xir.Graph.deserialize(model)
        subgraphs = get_child_subgraph_dpu(g)
        self.dpuRunners = []
        for i in range(maxNumThreads):
            self.dpuRunners.append(vart.Runner.create_runner(subgraphs[0], "run"))

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
        self.outputWrittenEvents[slot].set()
        # We free the handler:
        self.lock.release()

    def allocateSlot(self, inputBatch, waitForSpace=True):
        if waitForSpace:
            if self.areQueuesFull(blocking=True):
                self.queuesAreJustNotFullEvent.wait()

        self.lock.acquire()
        # self.runningThreadsSemaphore.acquire()
        if not waitForSpace:
            assert self.areQueuesFull(blocking=False)
        slot = self.firstEmptySlot
        self.threadsQueue[slot] = RunningThread(slot, inputBatch, self.dpuRunners[slot])
        self.outputBatchesQueue[slot] = None
        self.numRunningThreads = self.numRunningThreads + 1
        if self.firstEmptySlot >= self.maxNumThreads:
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
        self.threadsQueue[slot].join()
        self.threadsQueue[slot] = None
        self.numRunningThreads = self.numRunningThreads - 1
        if self.firstBusySlot >= self.maxNumThreads:
            self.firstBusySlot = 0
        else:
            self.firstBusySlot = self.firstBusySlot + 1
        # self.runningThreadsSemaphore.release()

        if self.numRunningThreads == (self.maxNumThreads - 1):
            self.queuesAreJustNotFullEvent.set()

        self.outputWrittenEvents[slot].clear()

        self.lock.release()


        return slot


    def areQueuesEmpty(self, blocking=True):
        if blocking:
            self.lock.acquire()
        numRunningThreads = self.numRunningThreads
        if blocking:
            self.lock.release()
        return numRunningThreads == 0 # and (self.firstEmptySlot == self.firstBusySlot)

    def areQueuesFull(self, blocking=True):
        if blocking:
            self.lock.acquire()
        numRunningThreads = self.numRunningThreads
        if blocking:
            self.lock.release()
        return numRunningThreads == self.maxNumThreads # and (self.firstEmptySlot == self.firstBusySlot)


