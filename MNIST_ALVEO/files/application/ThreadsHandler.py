import threading
from time import sleep


class RunningThread (threading.Thread):
    def __init__(self, handler, slot: int):
        threading.Thread.__init__(self)
        self.handler = handler
        self.slot = slot

    def run(self):
        inputBatch = self.handler.readInputBatch(self.slot)
        outputBatch = None
        # ...
        self.handler.writeOutputBatch(self.slot, outputBatch)

class DisplayThread (threading.Thread):
    def __init__(self, handler):
        threading.Thread.__init__(self)
        self.handler = handler

    def display(self, batch):
        # ...
        pass

    def run(self):
        busySlot = self.handler.getFirstBusySlot()
        while True:
            self.handler.outputWrittenEvents[busySlot].wait()

            batch = self.handler.outputBatchesQueue[busySlot]
            self.display(batch)

            self.handler.deallocateSlot()


class ThreadsHandler:
    def __init__(self, maxNumThreads: int):
        self.maxNumThreads = maxNumThreads
        self.firstBusySlot = 0
        self.firstEmptySlot = 0
        # self.numRunningThreads = 0
        self.runningThreadsSemaphore = threading.Semaphore(maxNumThreads)

        self.threadsQueue = [None] * maxNumThreads
        self.inputBatchesQueue = [None] * maxNumThreads
        self.outputBatchesQueue = [None] * maxNumThreads


        self.outputWrittenEvents = [threading.Event()] * maxNumThreads
        self.queuesAreJustNotFullEvent = threading.Event()
        self.lock = threading.Lock()

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

    def readInputBatch(self, slot):
        # We reserve the handler:
        self.lock.acquire()
        # We read the batch:
        batch = self.inputBatchesQueue[slot]
        # We free the handler:
        self.lock.release()
        return batch

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
        self.threadsQueue[slot] = RunningThread(slot)
        self.inputBatchesQueue[slot] = inputBatch
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
        self.inputBatchesQueue[slot] = None
        self.numRunningThreads = self.numRunningThreads - 1
        if self.firstBusySlot >= self.maxNumThreads:
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
        return numRunningThreads == 0 # and (self.firstEmptySlot == self.firstBusySlot)

    def areQueuesFull(self, blocking=True):
        if blocking:
            self.lock.acquire()
        numRunningThreads = self.numRunningThreads
        if blocking:
            self.lock.release()
        return numRunningThreads == self.maxNumThreads # and (self.firstEmptySlot == self.firstBusySlot)


