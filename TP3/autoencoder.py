import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

# ====================================================================================================================


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    # Redefinimos el método .__len__()

    def __len__(self):
        return len(self.dataset)
    # Redefinimos el método .__getitem__()

    def __getitem__(self, i):
        image, label = self.dataset[i]
        input = image
        # Reemplazamos el label original con una version achatada de la imagen. (matriz 28x28 en vector de 784)
        output = torch.flatten(image)
        return input, output

# ====================================================================================================================


def batch(x):
    return x.unsqueeze(0)  # (28,28) -> (1,28,28)


def unbatch(x):
    return x.squeeze().detach().cpu().numpy()  # (1,28,28) -> (28,28)


# Definimos la funcion de entrenamiento


def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Activamos la maquinaria de entrenamiento del modelo
    model.train()
    # Definimos ciertas constantes
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0
    model = model.to(device)
    # Iteramos sobre lotes (batchs)
    for batch, (X, y) in enumerate(dataloader):
        # Copiamos las entradas y las salidas al dispositivo de trabajo
        X = X.to(device)
        y = y.to(device)  # shape de (100, 784)
        # Calculamos la predicción del modelo y la correspondiente pérdida (error)
        pred = model(X)  # tiene shape de (100, 1, 28, 28)
        # aca hacemos reshape para que quede tambien de (100, 784)
        pred = pred.view(int(size/num_batches), 28*28)
        loss = loss_fn(pred, y)
        # Backpropagamos usando el optimizador proveido.
        optimizer.zero_grad()  # Reseteamos los gradientes a cero
        loss.backward()  # calculamos los gradientes con backpropagation
        optimizer.step()  # Actualizamos los pesos
        # Imprimimos el progreso...
        loss_value = loss.item()
        sum_loss += loss_value
        if batch % int(num_batches/6) == 0:
            current = batch*len(X)
            print(
                f"batch={batch} loss={loss_value:>7f}  muestras-procesadas:[{current:>5d}/{size:>5d}]")
    avg_loss = sum_loss/num_batches
    return avg_loss

# De manera similar, definimos la función de validación


def valid_loop(dataloader, model, loss_fn, device):
    # Desactivamos la maquinaria de entrenamiento del modelo
    model.eval()
    # Definimos ciertas constantes
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss = 0
    model = model.to(device)
    # Para testear, desactivamos el cálculo de gradientes.
    with torch.no_grad():
        # Iteramos sobre lotes (batches)
        for X, y in dataloader:
            # Copiamos las entradas y las salidas al dispositivo de trabajo
            X = X.to(device)
            y = y.to(device)
            # Calculamos las predicciones del modelo...
            pred = model(X)
            pred = pred.view(int(size/num_batches), 28*28)
            # y las correspondientes pérdidas (errores), los cuales vamos acumulando en un valor total.
            sum_loss += loss_fn(pred, y).item()
    # Calculamos la pérdida total y la fracción de clasificaciones correctas, y las imprimimos.
    avg_loss = sum_loss/num_batches
    print(f"Valid Error: Avg loss: {avg_loss:>8f} \n")
    return avg_loss

# ====================================================================================================================


def model_generator(NeuralNetwork, num_n, drop, num_epochs, num_batch, opti, learning_rate, train_data, valid_data):
    # Creamos los DataLoaders
    train_loader = DataLoader(train_data, batch_size=num_batch, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=num_batch, shuffle=True)

    # Creamos una instancia de una función de pérdida, una MSE loss en este caso
    loss_fn = nn.MSELoss()  # usar para autoencoder

    model = NeuralNetwork(n=num_n, p=drop)

    # Creamos un optimizador, un Stochastic Gradient Descent, en este caso.
    if opti == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif opti == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, eps=1e-08, weight_decay=0, amsgrad=False)

    # Determinamos en que dispositivo vamos a trabajar, una CPU o una GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Pasamos el modelo al dispositivo
    model = model.to(device)

    # 2.8) y 2.9)
    # Finalmente, entrenamos iterando sobre épocas.
    # Además, testeamos el modelo en cada una de ellas.
    num_epochs = num_epochs
    list_train_avg_loss = []
    list_valid_avg_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_avg_loss_incorrecta = train_loop(
            train_loader, model, loss_fn, optimizer, device)
        train_avg_loss = valid_loop(train_loader, model, loss_fn, device)
        valid_avg_loss = valid_loop(valid_loader, model, loss_fn, device)
        list_train_avg_loss.append(train_avg_loss)
        list_valid_avg_loss.append(valid_avg_loss)
    print("Done!")

    # plot
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(list(range(1, len(list_train_avg_loss) + 1)),
             list_train_avg_loss, label="train", linestyle='-', c='red')
    plt.plot(list(range(1, len(list_valid_avg_loss) + 1)),
             list_valid_avg_loss, label="valid", linestyle='-', c='blue')
    plt.title('')
    plt.legend()

    return model

# ====================================================================================================================


def test_model(model, test_data, indexes=None):
    figure = plt.figure()
    rows, cols = 3, 2
    i = 0  # subplot index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    if indexes == None:
        for row in range(1, rows+1):
            # Los números aleatorios tambien se pueden generar desde pytorch. Util para trabajar en la GPU.
            j = torch.randint(len(test_data), size=(1,)).item()
            image, flatten_img = test_data[j]
            image = image.to(device)
            # ploteamos la imagen original
            i += 1
            figure.add_subplot(rows, cols, i)
            if row == 1:
                plt.title("original")
            plt.axis("off")
            plt.imshow(unbatch(image), cmap="Greys_r")
            # ploteamos la imagen predicha
            i += 1
            figure.add_subplot(rows, cols, i)
            if row == 1:
                plt.title("predicha")
            plt.axis("off")
            image_pred = unbatch(model(batch(image)))
            plt.imshow(image_pred, cmap="Greys_r")
        plt.show()
    else:
        for row in range(1, rows+1):
            image, flatten_img = test_data[indexes[row-1].item()]
            image = image.to(device)
            # ploteamos la imagen original
            i += 1
            figure.add_subplot(rows, cols, i)
            if row == 1:
                plt.title("original")
            plt.axis("off")
            plt.imshow(unbatch(image), cmap="Greys_r")
            # ploteamos la imagen predicha
            i += 1
            figure.add_subplot(rows, cols, i)
            if row == 1:
                plt.title("predicha")
            plt.axis("off")
            image_pred = unbatch(model(batch(image)))
            plt.imshow(image_pred, cmap="Greys_r")
        plt.show()

# ====================================================================================================================
