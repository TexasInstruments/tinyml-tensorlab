# torch imports
import torch
from torch.ao.quantization import quantize_fx
import torch.utils
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, random_split
import torchinfo

# ti, onnx imports
from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule  # type: ignore
import onnx
import onnxruntime as ort

# other imports
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MotorFaultDataset(Dataset):
    """
    Create torch Dataset with windows of length window_length and offset of window_offset.
    The window will have a single target value which will be of the max occurring target.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, window_length: int, window_offset: int, data_format: str = 'NHWC') -> None:
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.y = torch.from_numpy(Y).type(torch.LongTensor)
        self.len = self.x.shape[0]
        # data points in a single window
        self.window_length = window_length
        # offset of data points not to overlap in a window
        self.window_offset = window_offset
        self.data_format = data_format

    def __getitem__(self, index: int):
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
        # find max occurring target value - we will use that as the y value for this window.
        window_samples_y = np.bincount(window_samples_y).argmax()
        return window_samples_x, window_samples_y

    def __len__(self):
        return (self.len-self.window_length)//self.window_offset


def get_dataset_from_csv(csv_file: str, normalize_dataset: bool = False) -> Tuple[np.ndarray]:
    """
    Read the csv_file to extract the X, Y values.
    normalize_dataset will normalize the column with the max value
    """
    df = pd.read_csv(csv_file)
    Y = df['Target'].to_numpy()
    if normalize_dataset:
        for column in df.columns:
            # normalize the values in a column with the max value
            df[column] = df[column] / df[column].abs().max()
    X = df[['Vibx', 'Viby', 'Vibz']].to_numpy()
    return X, Y


def get_dataloader(X: np.ndarray, Y: np.ndarray, window_length: int, window_offset: int, batch_size: int) -> Tuple[DataLoader]:
    """
    Get the torch dataloaders from the X, Y of the dataset. Create windows in dataset
    with length of window_length, offset of window_offset and batch size of dataloader.
    The function also shuffles the dataset after forming windows.
    """
    dataset = MotorFaultDataset(X, Y, window_length, window_offset)
    # split the dataset in training and testing
    train_dataset, test_dataset = random_split(dataset, lengths=[0.8, 0.2])
    # convert train and test dataset to torch dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer):
    """
    Train the model (torch model or qat wrapped torch model) with loss_fn, 
    optimizer on the torch train dataloader. Returns the avg loss for training 
    with model, loss_fn, optimizer
    """
    avg_loss = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        # make predictions for the current batch
        pred = model(X)
        pred = pred.flatten(start_dim=1)
        # compute the loss and its gradients
        loss = loss_fn(pred, y)
        loss.backward()
        # adjust the learning weights
        optimizer.step()
        # zero the gradients for every batch
        optimizer.zero_grad()
        avg_loss += loss.item()
    avg_loss = avg_loss/len(dataloader)
    return avg_loss, model, loss_fn, optimizer


def get_nn_model(in_channels: int, hidden_channels: List[int], feature_size: Tuple[int], out_channels: int, normalize_input: bool = True) -> nn.Module:
    """
    Get the torch model using the in_channels, hidden_channels, feature_size, out_channels
    The function will add the conv, bn, relu layers according to the hidden channels.
    AdaptiveAvgPool2D is added to reduce the dimension and finally a Linear layer at the last of
    the model
    """
    def get_conv_bn_relu(in_channels: int, out_channels: int, kernel_size: Tuple[int], padding=None, stride=1):
        # calculate the padding according to kernel if not provided
        padding = padding or (kernel_size[0]//2, kernel_size[1]//2)
        layers = []
        # perform conv, bn and relu on the input with in_channels and output of out_channels
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, padding=padding, stride=stride)]
        layers += [nn.BatchNorm2d(num_features=out_channels)]
        layers += [nn.ReLU()]
        return layers
    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = in_channels
            layers = []
            if normalize_input:
                # normalize the input with input features as in_channels
                layers += [nn.BatchNorm2d(num_features=in_channels)]
            else:
                layers += [nn.Identity()]

            # get the conv, bn, relu layers for each hidden channels
            in_ch = in_channels
            for h_ch in hidden_channels:
                layers += get_conv_bn_relu(in_ch, h_ch, kernel_size=(5, 1), padding=None, stride=(2, 1))
                in_ch = h_ch

            # reduces the dimensions of the nn layer to the output_size
            layers += [nn.AdaptiveAvgPool2d(output_size=feature_size)]

            # flatten the layer in last_hidden_layer*feature_size
            in_fc_ch = (in_ch*feature_size[0]*feature_size[1])
            layers += [nn.Flatten()] 

            # linearize the last layer in given out_features
            layers += [nn.Linear(in_fc_ch, out_features=out_channels)]

            # convert the layers in a pytorch understandable module list
            self.layers = nn.ModuleList(layers)

        def forward(self, x: torch.Tensor):
            for layer in self.layers:
                x = layer(x)
            return x

    nn_model = NeuralNetwork().to(DEVICE)
    return nn_model


def get_qat_model(nn_model: nn.Module, example_input: torch.Tensor, total_epochs: int) -> nn.Module:
    """
    Convert the torch model to qat wrapped torch model. The function requires 
    an example input to convert the model.
    """
    # Wrap the NN Model inside the QAT Wrapper
    import simple_qconfig
    qconfig_type = simple_qconfig.get_default_qconfig()
    qconfig_mapping = simple_qconfig.get_default_qconfig_mapping(qconfig_type)
    
    # QAT_model = quantize_fx.prepare_qat_fx(nn_model, qconfig_mapping, example_input)
    QAT_model = TINPUTinyMLQATFxModule(nn_model, qconfig_type=qconfig_type, example_inputs=example_input, total_epochs=total_epochs)
    return QAT_model


def train_model(model: nn.Module, dataloader: DataLoader, total_epochs: int, learning_rate: float) -> nn.Module:
    """
    Train the model (torch model or qat wrapped torch model) with the given train dataloader,
    total_epochs and a learning rate which will be used by lr_scheduler. CrossEntropyLoss and
    SGD are used as Loss Fn and optimizer to train.
    """
    # loss_fn for multi class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    # SGD is the preferable optimizer if QAT needs to be done
    opti = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.01, weight_decay=0.001)
    # vary the learning rate as per the lr_scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opti, total_epochs)

    for epoch in range(total_epochs):
        # train the model for an epoch
        loss, model, loss_fn, opti = train(dataloader, model, loss_fn, opti)
        # change the learning rate with scheduler
        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {epoch+1}\t LR: {round(last_lr,5)}\t Loss: {round(loss, 5)}")

    return model

def rename_input_node_for_onnx_model(onnx_model, input_node_name: str):
    """Rename the node of an ONNX model"""
    # Update graph input name.
    onnx_model.graph.input[0].name = input_node_name
    # Update input of the first node also to correspond.
    onnx_model.graph.node[0].input[0] = input_node_name
    # Check and write out the updated model
    onnx.checker.check_model(onnx_model)
    return onnx_model

def export_model(nn_model, example_input: torch.Tensor, model_name: str, with_qat: bool = False) -> nn.Module:
    """
    Export the model (torch model or qat wrapped torch model) to the given model name
    in the disk. The function requires an example input to save the model.
    """

    # Convert PyTorch QDQ layers to TI NPU int8 layers.
    nn_model.to(DEVICE)

    if with_qat:
        if hasattr(nn_model, "convert"):
            nn_model = nn_model.convert()
        else:
            nn_model = quantize_fx.convert_fx(nn_model.module)

    if hasattr(nn_model, "export"):
        # Export int8 quantized model to onnx.
        nn_model.export(example_input, model_name)
    else:
        torch.onnx.export(nn_model, example_input, model_name)

    # Set input name in the ONNX model to 'input' for consistency with float model
    load_onnx = onnx.load(model_name)
    updated_model = rename_input_node_for_onnx_model(load_onnx, 'input')
    # save the model in disk
    onnx.save(updated_model, model_name)
    return updated_model


def validate_model(model: nn.Module, test_loader: DataLoader, num_categories: int, categories_name: List[str]) -> float:
    """
    The function takes the model (torch model or qat wrapped torch model), torch dataloader
    and the num_categories to give the confusion matrix and accuracy of the model.
    """
    model.eval()
    y_target = []
    y_pred = []
    for _, (X, y) in enumerate(test_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        # make prediction for the current batch
        pred = model(X)
        pred = pred.flatten(start_dim=1)
        # take the max probability among the classes predicted
        _, pred = torch.max(pred, 1)
        y_pred.append(pred.numpy())
        y_target.append(y.numpy())

    y_pred = np.concatenate(y_pred)
    y_target = np.concatenate(y_target)
    categories_idx = np.arange(0, num_categories, 1)
    # create a confusion matrix
    cf_matrix = confusion_matrix(y_target, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[categories_name[i] for i in categories_idx],
                         columns=[categories_name[i] for i in categories_idx])
    print()
    print("Confusion Matrix")
    print(df_cm)

    # Accuracy of the model
    accuracy = np.diag(df_cm).sum()/np.array(df_cm).sum()
    return accuracy


def validate_saved_model(model_name: str, dataloader: DataLoader) -> float:
    """
    The function takes the saved onnx model, torch test dataloader to give the accuracy of the model.
    """
    correct_predictions = 0
    total_predictions = 0
    # set ort inference session options
    ort_session_options = ort.SessionOptions()
    ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # start the inference session with the model_name saved in disk
    ort_session = ort.InferenceSession(f"{model_name}", ort_session_options)

    for _, (X, Y) in enumerate(dataloader):
        for idx in range(len(X)):
            total_predictions += 1
            # add a new axis at the beginning of data
            data_point = X[idx].numpy()[np.newaxis, ...]
            # classify the data_point
            # outputs = ort_session.run(None, {'x': data_point})
            outputs = ort_session.run(None, {'input': data_point})
            conf_1 = outputs[0].flatten()
            # check if the classification is correct
            if conf_1.argmax(0) == Y[idx]:
                correct_predictions += 1

    accuracy = round(correct_predictions/total_predictions, 5)
    return accuracy

if __name__ == '__main__':

    MODEL_NAME = "motor_fault.onnx"
    CSV_FILE = "motor_fault_dataset.csv"
    CATEGORIES_NAME = ['Normal', 'Localized', 'Erosion', 'Flaking']
    NUM_EPOCHS = 25 #10
    WINDOW_LENGTH = 1024
    WINDOW_OFFSET = WINDOW_LENGTH//4  # WINDOW_LENGTH//2
    BATCH_SIZE = 64
    LEARNING_RATE = 0.1
    ENABLE_QAT = True  # False

    X, Y = get_dataset_from_csv(CSV_FILE)

    # number of columns in X to be trained
    IN_CHANNELS = X.shape[-1]
    # number of categories to be classified into
    NUM_CATEGORIES = len(np.unique(Y))
    assert len(CATEGORIES_NAME) == NUM_CATEGORIES, "Incorrect number of categories"
    print(f"Dataset: Samples={X.shape[0]}, Categories={NUM_CATEGORIES}")

    train_loader, test_loader = get_dataloader(X, Y, WINDOW_LENGTH, WINDOW_OFFSET, BATCH_SIZE)

    # get example input
    # dataloader returns a batch of input - take the first value output it to get single input for QAT config
    example_input, example_target = next(iter(train_loader))
    example_input = example_input[:1]

    nn_model = get_nn_model(IN_CHANNELS, hidden_channels=[8, 16, 32], feature_size=(4, 1), out_channels=NUM_CATEGORIES)
    torchinfo.summary(nn_model, input_data=example_input)

    nn_model = train_model(nn_model, train_loader, NUM_EPOCHS, LEARNING_RATE)
    accuracy = validate_model(nn_model, test_loader, NUM_CATEGORIES, CATEGORIES_NAME)
    export_model(nn_model, example_input, MODEL_NAME)
    print(f"Trained Model Accuracy: {round(accuracy, 5)}\n")

    if ENABLE_QAT:
        MODEL_NAME = 'qat_' + MODEL_NAME
        qat_epochs = max(NUM_EPOCHS//2, 5)
        qat_model = get_qat_model(nn_model, example_input=example_input, total_epochs=qat_epochs)
        
        qat_learning_rate = LEARNING_RATE/10
        qat_model = train_model(qat_model, train_loader, qat_epochs, qat_learning_rate)

        accuracy = validate_model(qat_model, test_loader, NUM_CATEGORIES, CATEGORIES_NAME)
        print(f"QAT Model Accuracy: {round(accuracy, 5)}\n")

        qat_model = export_model(qat_model, example_input, MODEL_NAME, with_qat=True)

    accuracy = validate_saved_model(MODEL_NAME, test_loader)
    print(f"Export ONNX QAT Model Accuracy: {round(accuracy, 5)}")
