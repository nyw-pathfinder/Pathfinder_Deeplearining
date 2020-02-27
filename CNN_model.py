import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class CNN(nn.Module): #CNN이라는 모델을 언제든지 만들 수 있도록 class를 지정하겠다. 이때 nn.module torch.nn.에서 제공하는 nn.module이라는 클래스를 가져와서 CNN클래스에도 상속하겠다는거다.
                      #nn. Module에는 backpropagation과 gradient구하는 기능이 있다.
    def __init__(self): # init은 class에서 인스턴스를 만들때 자동으로 생성되는 것.
        super(CNN, self).__init__()  #CNN가 상속받는 class는 nn.Module을 상속받는다. init의 상속을 보장해준다.
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), #커널에 튜플()을 사용하면 3*3인거 나타낸다.
                                     padding=1)  # 1x28x28((MNIST는 흑백이라1/ MINIT는 28*28사이즈다.) -> 32x28x28 아웃풋은 32고 여기 28이이 layer 1로 넘어간다.
        self.layer_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)  # 64x14x14 #[28-3+2]/2 = 14 이 공식은 feature의 세로 가로로.
        self.layer_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)  # 128x7x7
        #채널수를 늘어난다? 채널 하나하나가 필터역할이다.  -> 필터가 늘어난다. edge 발견할게 많아진다. 채널 생길때마다 random edge부여됨. 물론 parameter가 늘어나서 오버피팅 될 수도 있지만 채널수 늘리고 세로가로 줄이는게 더 났다.
        ##!  bias는 어디에 있는거가. 필터에 이미 포함되어 있는건가....
        self.layer_3 = nn.AdaptiveAvgPool2d((1, 1))  # 128x1x1  dimention을 줄여준다. 128개의 채널 전체 평균으로 나눠 버린다. why? FC layer에서 classcifiction 할거니까.
        self.layer_4 = nn.Linear(in_features=128,
                                 out_features=10)  # 10. FC layer로 이미 데이터가 1by1으로 바뀌었고 classofocation 하기 좋다.
        # self.act = nn.ReLU() activation은 모델의 표현력 증가시키는데 AVgpool은 형상을 바꾼다.
        # softmax가 이미 loss 함수에 들어있다. 파이토치에선

    def forward(self, x):
        x1 = F.relu(self.input_layer(x))  # 16x28x28 #activation function이
        x2 = F.relu(self.layer_1(x1))  # 32x14x14
        x3 = F.relu(self.layer_2(x2))  # 64x7x7
        x4 = self.layer_3(x3)  # Bx64x1x1
        x5 = x4.view(x4.shape[0], -1)  ##! 이녀석 의미를 잘 모르겠다.
        output = self.layer_4(x5)  # Bx10
        return output


transforms = Compose([ToTensor(),  # -> [0,1] #전체로 나눠준다. 원래 음수였던 친구는 - 친구들이 0으로 간다. 순서는 0에서1을 -0.5로 평행이동 후 나눠준다. data 가져올때 여기서 transform해서 가져와라.
                      Normalize(mean=[0.5], std=[0.5])])  # -> [-1,1] #-1에서 1로.
dataset = MNIST(root='./datasets', train=True, transform=ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True) #64개씩 준다.배치사이즈 순서 조정해서 for문에 준다.

model = CNN() #이 괄호는 CNN이 class 라는 걸 의미. 쇗물 부어넣는 작업. CNN인스턴스를 만들겠다.라고 ()가 있다고 인식.

criterion = nn.CrossEntropyLoss() #잘했냐 잘 못했냐 척도 알려줌.

optim = torch.optim.Adam(model.parameters(), lr=0.001)  # weight_new = weight_old - weight_gradient * lr 횟초리녀석. weight를 때린다.


list_loss = []
list_acc = []
for epoch in range(1):
    for input, label in tqdm(data_loader):
        # label 32 왜 여기서 32 라고 적어놨는지 잘 모르겠다. 위에서 64개씩 준다고 했는데 왜 32인가?
        output = model(input)  # 32x10
        loss = criterion(output, label)  # 1 #결과, 라벨 순서 꼭 지켜야함 안지키면 돌아가긴 함 근데 결과가 전혀 다름

        optim.zero_grad()#잘못한 이전의 그라디언트를 0으로 만들어줌
        loss.backward()#웨이트들이게 그라디언트를 돌려줌
        optim.step()  # 여기서 파라미터들의 값이변함 (위 세 줄의 순서가 중요)
        list_loss.append(loss.detach().item()) #파이토치 실수형을 파이썬 실수형으로 바꿔준다. GPU기반데이터라 파이토치 자료형은 조금 다르기 때

        n_correct_answers = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item()  ##! 무슨의미인지 잘 모르겠다.
        print("Accuracy: ", n_correct_answers / 32.0 * 100) ##! 왜 32로 나누는가.
        list_acc.append(n_correct_answers / 32.0 * 100)


plt.plot(list_loss)
plt.plot(list_acc)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

#이 코드를 돌렸을 때 왜 accuracy가 100을 넘어가는가. 원래 100까지만 나와야 하지 않나?
