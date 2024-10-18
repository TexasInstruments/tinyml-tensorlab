# torch imports
import enum
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.utils
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchinfo

# ti, onnx imports
import tinyml_torchmodelopt.quantization as tinpu_quantization  # type: ignore
import onnx
import onnxruntime as ort

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MotorFaultDataset(Dataset):
    def __init__(self, X, Y, window_length, window_offset, data_format='NHWC'):
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.y = torch.from_numpy(Y).type(torch.LongTensor)
        self.len = self.x.shape[0]
        self.window_length = window_length
        self.window_offset = window_offset
        self.data_format = data_format

    def __getitem__(self, index):
        start_offset = index*self.window_offset
        # form a window of input values
        window_samples_x = self.x[start_offset:start_offset+self.window_length]
        if self.data_format == 'NHWC':
            # neural network expects in NCHW, so transpose the data
            window_samples_x = np.transpose(window_samples_x, (1, 0))
        if len(window_samples_x.shape) == 2:
            window_samples_x = window_samples_x[..., np.newaxis]

        # form a window of target values
        window_samples_y = self.y[start_offset:start_offset+self.window_length]
        # find max occuring target value - we will use that as the y value for this window.
        window_samples_y = np.bincount(window_samples_y).argmax()
        return window_samples_x, window_samples_y

    def __len__(self):
        return (self.len-self.window_offset)//self.window_offset


def get_dataset_from_csv(csv_file, normalize_dataset=False):
    import pandas as pd
    df = pd.read_csv(csv_file)
    Y = df['Target'].to_numpy()
    if normalize_dataset:
        for column in df.columns:
            df[column] = df[column] / df[column].abs().max()
    X = df[['Vibx', 'Viby', 'Vibz']].to_numpy()
    return X, Y


def get_dataloader_from_dataset(X, Y, window_length, window_offset, batch_size):
    dataset = MotorFaultDataset(X, Y, window_length, window_offset)
    train_dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer):
    avg_loss = 0
    model.train()
    for idx, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        pred = pred.flatten(start_dim=1)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.item()
    avg_loss = avg_loss/len(dataloader)
    return avg_loss, model, loss_fn, optimizer


def get_nn_model(in_channels, hidden_channels, out_channels, feature_size=(1,1), normalize_input=True):
    def get_conv_bn_relu(in_channels, out_channels, kernel_size, padding=None, stride=1):
        padding = padding or (kernel_size[0]//2, kernel_size[1]//2)
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, padding=padding, stride=stride)]
        layers += [nn.BatchNorm2d(num_features=out_channels)]
        layers += [nn.ReLU()]
        return layers

    class ReshapeLayer(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = in_channels
            layers = []
            if normalize_input:
                layers = [nn.BatchNorm2d(num_features=in_channels)]
            in_ch = in_channels
            for h_ch in hidden_channels:
                layers += get_conv_bn_relu(in_ch, h_ch,
                                           kernel_size=(5, 1), padding=None, stride=(2, 1))
                in_ch = h_ch
            layers += [nn.AdaptiveAvgPool2d(output_size=feature_size)]
            layers += [ReshapeLayer()]
            in_fc_ch = (in_ch*feature_size[0]*feature_size[1])
            layers += [nn.Linear(in_fc_ch, out_features=out_channels)]
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            x = self.layers(x)
            return x

    nn_model = NeuralNetwork().to(DEVICE)
    return nn_model


def get_qat_model(nn_model, example_input, total_epochs):
    qconfig_type = {
        'weight': {
            'bitwidth': 8,
            'qscheme': torch.per_channel_symmetric,
            'power2_scale': True
        },
        'activation': {
            'bitwidth': 8,
            'qscheme': torch.per_tensor_symmetric,
            'power2_scale': True,
            'range_max': None,
            'fixed_range': False
        }
    }

    # Define the QAT model
    QAT_model = tinpu_quantization.TINPUTinyMLQATFxModule(
        nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)

    return QAT_model


def train_model(model, dataloader, total_epochs, learning_rate):
    loss_fn = torch.nn.CrossEntropyLoss()
    opti = torch.optim.SGD(params=model.parameters(),
                           lr=learning_rate, momentum=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opti, total_epochs)
    for epoch in range(total_epochs):
        loss, model, loss_fn, opti = train(dataloader, model, loss_fn, opti)
        scheduler.step()
        print(f"Epoch:  {epoch+1}  Loss :  {round(loss, 5)}")
    return model


def export_model(nn_model, example_input, model_name):

    def rename_input_node_for_onnx_model(onnx_model, input_node_name):
        """Rename the node of an ONNX model"""
        # Update graph input name.
        onnx_model.graph.input[0].name = input_node_name
        # Update input of the first node also to correspond.
        onnx_model.graph.node[0].input[0] = input_node_name
        # Check and write out the updated model
        onnx.checker.check_model(onnx_model)
        return onnx_model

    # Convert PyTorch QDQ layers to TI NPU int8 layers.
    nn_model.to(DEVICE)

    if hasattr(nn_model, "convert"):
        nn_model = nn_model.convert()

    if hasattr(nn_model, "export"):
        # Export int8 quantized model to onnx.
        nn_model.export(example_input, model_name)
    else:
        torch.onnx.export(nn_model, example_input, model_name)

    # Set input name in the ONNX model to 'input' for consistency with float model
    load_onnx = onnx.load(model_name)
    updated_model = rename_input_node_for_onnx_model(load_onnx, 'input')
    onnx.save(updated_model, model_name)
    return updated_model


def validate_saved_model(model_name, dataloader):
    correct_predictions = 0
    total_predictions = 0
    ort_session_options = ort.SessionOptions()
    ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ort_session = ort.InferenceSession(f"{model_name}", ort_session_options)

    for _, (X, Y) in enumerate(dataloader):
        for idx in range(len(X)):
            total_predictions += 1
            data_point = X[idx].numpy()[np.newaxis, ...]
            outputs = ort_session.run(None, {"input": data_point})
            conf_1 = outputs[0].flatten()
            if conf_1.argmax(0) == Y[idx]:
                correct_predictions += 1

    accuracy = round(correct_predictions/total_predictions, 5)
    return accuracy


def validate_model(model, test_loader):
    import pandas as pd

    model.eval()
    y_target = []
    y_pred = []
    for _, (X, y) in enumerate(test_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        pred = pred.flatten(start_dim=1)
        _, pred = torch.max(pred, 1)
        y_pred.append(pred.cpu().numpy())
        y_target.append(y.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    y_target = np.concatenate(y_target)
    classes = np.arange(0, 4, 1)
    cf_matrix = confusion_matrix(y_target, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    print("Confusion Matrix")
    print(df_cm)

    # Accuracy of the model
    accuracy = np.diag(df_cm).sum()/np.array(df_cm).sum()
    return accuracy


if __name__ == '__main__':

    MODEL_NAME = "motor_fault.onnx"
    CSV_FILE = "motor_fault_dataset.csv"
    NUM_EPOCHS = 50
    WINDOW_LENGTH = 1024
    WINDOW_OFFSET = WINDOW_LENGTH//2
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    ENABLE_QAT = True  # False

    X, Y = get_dataset_from_csv(CSV_FILE)
    in_channels = X.shape[-1]
    num_categories = len(np.unique(Y))
    print(
        f"Dataset: Num samples={X.shape[0]}, Num categories={num_categories}")

    train_loader, test_loader = get_dataloader_from_dataset(
        X, Y, WINDOW_LENGTH, WINDOW_OFFSET, BATCH_SIZE)

    # get example input
    # dataloader returns a batch of input - take the first value output it to get single input
    example_input, example_target = next(iter(train_loader))
    example_input = example_input[:1]

    nn_model = get_nn_model(in_channels, 
            hidden_channels=[8, 16, 32, 64], out_channels=num_categories)
    torchinfo.summary(nn_model, input_data=example_input)

    nn_model = train_model(nn_model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    accuracy = validate_model(nn_model, test_loader)
    export_model(nn_model, example_input, MODEL_NAME)
    print("Trained Model Accuracy: ", round(accuracy, 5))

    if ENABLE_QAT:
        qat_epochs = max(NUM_EPOCHS//2, 5)
        qat_model = get_qat_model(
            nn_model, example_input=example_input, total_epochs=qat_epochs)
        qat_learning_rate = LEARNING_RATE/10
        qat_model = train_model(qat_model, train_loader,
                                qat_epochs, qat_learning_rate)
        accuracy = validate_model(qat_model, test_loader)
        print("QAT Model Accuracy: ", round(accuracy, 5))
        export_model(qat_model, example_input, MODEL_NAME)

    accuracy = validate_saved_model(MODEL_NAME, test_loader)

    print("Export ONNX QAT Model Accuracy: ", round(accuracy, 5))
    exit()
