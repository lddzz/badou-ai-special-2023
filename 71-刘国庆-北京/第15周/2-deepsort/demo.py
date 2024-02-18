# 导入时间模块,用于计算处理时间
import time
# 导入OpenCV库,用于处理图像和视频
import cv2
# 从dcmtracking库中导入Yolov5DeepSortTracker类,用于目标检测和跟踪
from dcmtracking.deep_sort.tracker.yolov5_deep_sort_tracker import Yolov5DeepSortTracker


# 函数deal_one_video:输入参数为Yolov5DeepSortTracker对象det,视频路径video_path,目标路径target_path
def deal_one_video(det, video_path, target_path):
    # 使用OpenCV打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 获取视频的帧率
    fps = int(cap.get(5))
    # 打印帧率
    print(f"帧率:{fps}")
    # 初始化帧计数器
    i = -1
    # 设置跳帧数
    skip = 1
    # 记录开始时间
    startTime = time.time()
    # 初始化时间统计字典
    tt_dict = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    # 获取视频的宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频的高度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 设置视频编码格式:mp4v,即MP4格式
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    # 创建视频写入对象
    # cv2.VideoWriter()：这是OpenCV库中的一个函数，用于创建一个VideoWriter对象，该对象可以将图像序列写入视频文件。
    # target_path：这是一个字符串，表示要写入的视频文件的路径。
    # fourcc：这是一个表示视频编码的四字符代码。'mp4v'代表MP4编码。
    # fps：这是一个整数，表示视频的帧率，即每秒钟的帧数。
    # (frame_width, frame_height)：这是一个元组，表示视频帧的宽度和高度
    videoWriter = cv2.VideoWriter(target_path, fourcc, fps, (frame_width, frame_height))
    # 循环处理每一帧
    while True:
        # 读取一帧
        _, im = cap.read()
        # 如果读取失败,跳出循环
        if im is None:
            break
        # 帧计数器加一
        i += 1
        # 如果当前帧需要处理
        if i % skip == 0:
            need_detect = True
        # 如果当前帧不需要处理
        else:
            need_detect = False
        # 调用Yolov5DeepSortTracker对象的deal_one_frame方法处理一帧
        # det.deal_one_frame()：返回处理后的图像im，检测到的目标的ID列表ids，以及对应的边界框列表bboxes
        # im：表示当前帧的图像。
        # fps：表示视频的帧率，即每秒钟的帧数。
        # need_detect：表示是否需要在当前帧上进行目标检测。
        im, ids, bboxes = det.deal_one_frame(im, fps, need_detect=need_detect)
        # 每处理100帧,打印一次处理时间和耗时统计
        if i % 100 == 0:
            print(f"第{i}帧:处理时间:{time.time() - startTime},耗时统计:{det.cost_dict}")
        # 记录开始写入时间
        startWriteTime = time.time()
        # 将处理后的帧写入视频文件
        videoWriter.write(im)
        # 计算写入耗时
        tt_dict["4"] += time.time() - startWriteTime
    # 释放视频文件
    cap.release()
    # 释放视频写入对象
    videoWriter.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    # 打印总耗时和各部分耗时统计
    print(f"总耗时:{time.time() - startTime},耗时统计:{det.cost_dict}")


# demo_yolov5_deep_sort_tracker:输入参数为视频路径和目标路径
def demo_yolov5_deep_sort_tracker(video_path, target_path):
    # 创建一个Yolov5DeepSortTracker对象,需要计算速度,不需要计算角度
    det = Yolov5DeepSortTracker(need_speed=True, need_angle=False)
    # 调用deal_one_video函数处理视频
    deal_one_video(det, video_path, target_path)


# 主函数
if __name__ == '__main__':
    # 调用demo_yolov5_deep_sort_tracker函数处理视频
    demo_yolov5_deep_sort_tracker('data/test5.mp4', 'data/out5.mp4')
    # 打印"处理完成"表示处理完成
    print("处理完成")
