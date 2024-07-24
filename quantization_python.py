import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.quantization


# Normalize values for CIFAR-10 (mean and std of dataset)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)

        ## xl
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        ## xl
        x = self.quant(x)

        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool2(nn.ReLU()(self.bn3(self.conv3(x))))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = self.pool3(nn.ReLU()(self.bn5(self.conv5(x))))
        x = nn.ReLU()(self.bn6(self.conv6(x)))
        x = x.contiguous().reshape(-1, 256 *  4 * 4)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        ## xl
        x = self.dequant(x)
        return x

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedNet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler for learning rate
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Training function
def train(model, train_loader, criterion, optimizer, scheduler, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Accuracy: {100. * correct / len(train_loader.dataset)}%')

# Train the model for more epochs
train(model, train_loader, criterion, optimizer, scheduler, epochs=2)


# Evaluate quantized model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Accuracy (No Quantized): {accuracy:.2f}%')

evaluate_model(model, test_loader)

## xl

# Quantization process (optional and only if needed)

# define a floating point model
model_fp32 = model
# model must be set to eval mode for static quantization logic to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'x86' for server inference and 'qnnpack'
# for mobile inference. Other quantization configurations such as selecting
# symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
# can be specified here.
# Note: the old 'fbgemm' is still available but 'x86' is the recommended default
# for server inference.
# model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
#model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
## 我不清楚上述代码会不会报错，报错了再告诉我改
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [
  ['conv1', 'bn1'],
  ['conv2', 'bn2'],
  ['conv3', 'bn3'],
  ['conv4', 'bn4'],
  ['conv5', 'bn5'],
  ['conv6', 'bn6']
])

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(4, 3, 32, 32).to(device)
model_fp32_prepared(input_fp32)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)


# run the model, relevant calculations will happen in int8
res = model_int8(input_fp32)
res = res.contiguous().reshape(res.size(0), -1)  # Ensure tensor is contiguous before reshaping
print(res)




# Evaluate quantized model
def evaluate_quantized_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Accuracy (Quantized): {accuracy:.2f}%')



# Use the quantized model for evaluation
evaluate_quantized_model(model_int8, test_loader)
# 将量化后的模型转换为TorchScript
traced_model = torch.jit.trace(model_int8, input_fp32)
# 保存TorchScript模型
torch.jit.save(traced_model, 'quantized_model.pth')
# 加载TorchScript模型
loaded_model = torch.jit.load('quantized_model.pth')
# 确保模型在评估模式下
loaded_model.eval()

# 运行加载的模型
res_loaded = loaded_model(input_fp32)
