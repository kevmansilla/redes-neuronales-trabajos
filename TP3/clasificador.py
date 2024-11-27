import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import seaborn as sns
import copy
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report
)

# ====================================================================================================================

# Definimos la función de entrenamiento


def train_loop_classifier(dataloader, model, loss_fn, optimizer, device):
    # Activamos la maquinaria de entrenamiento del modelo
    model.train()
    # Definimos ciertas constantes
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss, sum_correct = 0, 0
    # Iteramos sobre lotes (batchs)
    for batch, (X, y) in enumerate(dataloader):
        # Copiamos las entradas y las salidas al dispositivo de trabajo
        X = X.to(device)
        y = y.to(device)
        # Calculamos la predicción del modelo y la correspondiente pérdida (error)
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagamos usando el optimizador proveido.
        optimizer.zero_grad()  # seteamos los valores de los gradientes a cero
        loss.backward()  # calculamos los gradientes con back propagation
        optimizer.step()  # usamos los valores de los gradientes para actualizar los pesos
        # Imprimimos el progreso...
        loss_value = loss.item()
        sum_loss += loss_value
        # También calculamos el número de predicciones correctas, y lo acumulamos en un total.
        sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            current = batch*len(X)
            print(
                f"batch={batch} loss={loss_value:>7f}  muestras-procesadas:[{current:>5d}/{size:>5d}]")
    avg_loss = sum_loss/num_batches
    frac_correct = sum_correct/size
    return avg_loss, frac_correct

# De manera similar, definimos la función de prueba


def valid_loop_classifier(dataloader, model, loss_fn, device):
    # Desactivamos la maquinaria de entrenamiento del modelo
    model.eval()
    # Definimos ciertas constantes
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_loss, sum_correct = 0, 0
    # Para testear, desactivamos el cálculo de gradientes.
    with torch.no_grad():
        # Iteramos sobre lotes (batches)
        for X, y in dataloader:
            # Copiamos las entradas y las salidas al dispositivo de trabajo
            X = X.to(device)
            y = y.to(device)
            # Calculamos las predicciones del modelo...
            pred = model(X)
            # y las correspondientes pérdidas (errores), los cuales vamos acumulando en un valor total.
            sum_loss += loss_fn(pred, y).item()
            # También calculamos el número de predicciones correctas, y lo acumulamos en un total.
            sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Calculamos la pérdida total y la fracción de clasificaciones correctas, y las imprimimos.
    avg_loss = sum_loss/num_batches
    frac_correct = sum_correct/size
    print(
        f"Test Error: \n Accuracy: {(100*frac_correct):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return avg_loss, frac_correct

# ====================================================================================================================


def classifier_generator(NeuralNetwork, num_n, trained_model, drop, num_epochs, num_batch, opti, learning_rate, train_data, valid_data):
    # Creamos los DataLoaders
    train_loader = DataLoader(train_data, batch_size=num_batch, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=num_batch, shuffle=True)

    # Creamos una instancia de una función de pérdida, una MSE loss en este caso
    loss_fn = nn.CrossEntropyLoss()  # usar para clasificacion

    model = NeuralNetwork(n=num_n, encoder=trained_model.encoder, p=drop)

    # Creamos un optimizador, un Stochastic Gradient Descent, en este caso.
    if opti == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif opti == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, eps=1e-08, weight_decay=0, amsgrad=False)
    elif opti == 'Adam classifier':
        optimizer = torch.optim.Adam(model.classifier.parameters(
        ), lr=learning_rate, eps=1e-08, weight_decay=0, amsgrad=False)
    elif opti == 'SGD classifier':
        optimizer = torch.optim.SGD(
            model.classifier.parameters(), lr=learning_rate)

    # Determinamos en que dispositivo vamos a trabajar, una CPU o una GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Pasamos el modelo al dispositivo
    model = model.to(device)

    # Finalmente, entrenamos iterando sobre épocas.
    # Además, testeamos el modelo en cada una de ellas.
    num_epochs = num_epochs
    list_train_avg_loss = []
    list_valid_avg_loss = []
    list_train_acc = []
    list_valid_acc = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        _loss, _acc = train_loop_classifier(
            train_loader, model, loss_fn, optimizer, device)
        train_avg_loss, train_acc = valid_loop_classifier(
            train_loader, model, loss_fn, device)
        valid_avg_loss, valid_acc = valid_loop_classifier(
            valid_loader, model, loss_fn, device)
        list_train_avg_loss.append(train_avg_loss)
        list_valid_avg_loss.append(valid_avg_loss)
        list_train_acc.append(train_acc)
        list_valid_acc.append(valid_acc)
    print("Done!")

    plt.xlabel('epoch')
    plt.ylabel('CEL')
    plt.plot(range(1, len(list_train_avg_loss) + 1),
             list_train_avg_loss, label="train", linestyle='-', c='red')
    plt.plot(range(1, len(list_valid_avg_loss) + 1),
             list_valid_avg_loss, label="valid", linestyle='-', c='blue')
    plt.title('')
    plt.legend()
    plt.show()

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(range(1, len(list_train_acc) + 1), list_train_acc,
             label="train", linestyle='-', c='red')
    plt.plot(range(1, len(list_valid_acc) + 1), list_valid_acc,
             label="valid", linestyle='-', c='blue')
    plt.title('')
    plt.legend()
    plt.show()
    # 3.10)
    matriz_confusion(model, valid_loader, device)

    return model

# ====================================================================================================================


def matriz_confusion(model, test_data, device):
    pred_label = []
    true_label = []
    model = model.to(device)
    model.eval()
    for inputs, labels in test_data:
        inputs = inputs.to(device)
        labels = torch.tensor(labels, dtype=torch.int8)
        labels = labels.to(device)
        outputs = model(inputs)

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        pred_label.extend(outputs)

        labels = labels.data.cpu().numpy()
        true_label.extend(labels)

    target_name = (
        "T-Shirt",    # Remera manga corta
        "Trouser",    # Pantalon
        "Pullover",   # Buzo
        "Dress",      # Vestido
        "Coat",       # Abrigo
        "Sandal",     # Sandalia
        "Shirt",      # Remera manga larga
        "Sneaker",    # Zapatilla
        "Bag",        # Bolso
        "Ankle Boot"  # Bota
    )

    print(classification_report(true_label, pred_label, target_names=target_name))
    ConfusionMatrixDisplay.from_predictions(
        true_label, pred_label, display_labels=target_name, xticks_rotation=45)
    plt.tight_layout()
    plt.show()
