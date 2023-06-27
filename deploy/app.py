# only used if script is run as 'main' from command line
import argparse
from ThreadsHandler import ThreadsHandler, DisplayThread
import cv2
import numpy as np

def app(source: str, threads: int, modelFile: str, targetFPS: int, batchSize: int = 1):
    # 1) The ThreadsHandler is initialized:
    handler: ThreadsHandler = ThreadsHandler(threads, modelFile)

    # 2) The DisplayThread is initialized:
    displayThread: DisplayThread = DisplayThread(handler, targetFPS)

    # 3) A VideoCapture is initialized:
    if(source[0].isdigit() and len(source) == 1):
        cameraId = int(source)
        video = cv2.VideoCapture(cameraId)
    else:
        video = cv2.VideoCapture(source)

    if not video.isOpened():
        print("Could not open ", source)
        return 1

    # 4) The video is run:
    batch = []
    displayThread.start()
    while (True):

        # Capture the video frame
        ret, frame = video.read()

        # Display the resulting frame
        # cv2.imshow('Input', frame)

        batch.append(frame)
        if len(batch) == batchSize:
            # Insert the batch into the handler:
            handler.allocateSlot(np.array(batch))
            batch = []

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video.release()

    # Destroy all the windows
    cv2.destroyAllWindows()



def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument('-s', '--source', type=str, default='0', help='Input video device ID or path to video. Default is 0 (webcam)')
    ap.add_argument('-s', '--source', type=str, default='video.mp4',help='Input video device ID or path to video. Default is 0 (webcam)')
    ap.add_argument('-t', '--threads', type=int, default=1, help='Number of threads. Default is 1')
    # ap.add_argument('-m', '--model', type=str, default='../vitis/compiled_pytorch_xmodel/Sequential_int.xmodel',
    #                help='Path of xmodel. Default is ../vitis/compiled_pytorch_xmodel/Sequential_int.xmodel')
    ap.add_argument('-m', '--model', type=str, default='../trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth',
                    help='Path of xmodel. Default is ../trt_pose/models/resnet18_baseline_att_224x224_A_epoch_249.pth')
    # ap.add_argument('-m', '--model', type=str,
    #                 default='../trt_pose/models/densenet121_baseline_att_256x256_B_epoch_160.pth',
    #                help='Path of xmodel. Default is ../trt_pose/models/densenet121_baseline_att_256x256_B_epoch_160.pth')

    ap.add_argument('-f', '--fps', type=int, default=24, help='Target FPS. Default is 24')
    ap.add_argument('-b', '--batchSize', type=int, default=1, help='Batch size. Default is 1')

    args = ap.parse_args()

    print('Command line options:')
    print(' --source : ', args.source)
    print(' --threads   : ', args.threads)
    print(' --model     : ', args.model)
    print(' --fps     : ', args.fps)
    print(' --batchSize     : ', args.batchSize)

    app(args.source, args.threads, args.model, args.fps, args.batchSize)


if __name__ == '__main__':
    main()