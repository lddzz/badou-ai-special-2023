import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from alexnet_model import AlexNet
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ]
    )
    # 加载图片
    image_path = './daisy2.jpg'
    assert os.path.exists(image_path),"file:'{}'dose not exist.".format(image_path)
    img = Image.open(image_path)

    plt.imshow(img)
    # [N,C,H,W]
    img = data_transform(img)
    # expand batch dimension扩展批处理维度
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path),"file:'{}' dose not exist.".format(json_path)
    with open(json_path,'r')as f:
        class_indict = json.load(f)
    # create model
    model = AlexNet(num_classes=5).to(device)

    # 加载模型权重
    weights_path = './Alexnet.pth'
    assert os.path.exists(weights_path),"file:'{}'dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class:{} proab；{:.3}".format(class_indict[str(predict_cla)],
                                             predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class:{:10} prob:{:.3}".format(class_indict[str(i)],predict[i].numpy()))
    plt.show()
if __name__=='__main__':
    main()