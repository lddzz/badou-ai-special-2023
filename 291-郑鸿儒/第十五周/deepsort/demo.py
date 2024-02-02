import cv2
import time
from dcmtracking.deep_sort.tracker.yolov5_deep_sort_tracker import Yolov5DeepSortTracker
from dcmtracking.deep_sort.tracker.yolo_fastestv2_deep_sort_tracker import YoloFastestV2DeepSortTracker
import imutils


def demo_yolov5_deep_sort_tracker(video_path, target_path):
    det = Yolov5DeepSortTracker(need_speed=True, need_angle=False)
    deal_one_video(det, video_path, target_path)

def demo_yolo_fastestv2_deep_sort_tracker(video_path, target_path):
    det = YoloFastestV2DeepSortTracker(need_speed=True, need_angle=False)
    deal_one_video(det, video_path, target_path)

def deal_one_video(det, video_path, target_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000 / fps)
    i = -1
    skip = 1
    tt = time.time()
    tt_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(


        'F', 'L', 'V', '1')
    videoWriter = cv2.VideoWriter(
        target_path, fourcc, fps, (frame_width, frame_height))
    while True:
        _, im = cap.read()
        if im is None:
            break
        i += 1
        if i % skip == 0:
            need_detect = True
        else:
            need_detect = False
        im, ids, bboxes = det.deal_one_frame(im, fps, need_detect=need_detect)
        if i % 100 == 0:
            print('i:', i, time.time() - tt, det.cost_dict)

        t1 = time.time()
        videoWriter.write(im)
        tt_dict['4'] += time.time() - t1
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    print('cost:', i, time.time() - tt, tt_dict, det.cost_dict)


if __name__ == '__main__':
    demo_yolov5_deep_sort_tracker('data/test5.mp4', 'data/out5.flv')
    print('Finished')
