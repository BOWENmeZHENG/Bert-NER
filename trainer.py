import torch
import torch.nn as nn
import utils_train as ut
import matplotlib.pyplot as plt
from datetime import date
import os
import numpy as np
import pandas as pd
import json

def train(model, tokenizer, train_json, test_json, classes, 
          n_data, batch_size, seed, max_length, class_weights: list, lr, n_epochs,
          plot=True, save_model=True, save_results=True):
    folder = f'{date.today()}_n_{n_data}_b_{batch_size}_s_{seed}_l_{max_length}_w_{class_weights}_l_{lr}'
    os.makedirs(f'saved_models/{folder}', exist_ok=True)

    data_batches, target_batches, att_mask_batches = ut.preprocess(json_file=train_json, classes=classes, tokenizer=tokenizer, 
                                                                   n_data=n_data, batch_size=batch_size, max_length=max_length, test=False)
    weights = torch.tensor(class_weights)
    weights_n = weights / torch.norm(weights)
    weights_n = torch.cat((weights_n, torch.tensor([0])))  # weights for padding = 0
    criterion = nn.CrossEntropyLoss(weight=weights_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    if test_json != None:
        test_accuracies = []
    for epoch in range(n_epochs):
        epoch += 1
        train_loss_batch = []
        train_accuracy_batch = []
        for b, X in enumerate(data_batches):
            y_pred = model(X, attention_mask=att_mask_batches[b])
            y_pred = torch.swapaxes(y_pred, 1, 2)
            y = target_batches[b]
            
            loss = criterion(y_pred, y)
            acc, *_ = ut.accuracy(0, len(classes), y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_batch.append(loss.item())
            train_accuracy_batch.append(acc)
        
        train_loss_batch_mean = sum(train_loss_batch) / len(train_loss_batch)
        train_accuracy_batch_mean = sum(train_accuracy_batch) / len(train_accuracy_batch)
        print(f'Epoch {epoch}')    
        print(f'Mean training loss: {train_loss_batch_mean:.4f}')
        print(f'Mean training accuracy: {train_accuracy_batch_mean:.4f}')
        if test_json != None:
            acc_test, pred_classes, true_classes, pred_all, true_all, data_list = testing(model, test_json, classes, tokenizer, max_length)
            test_accuracies.append(acc_test)
            print(f'Mean test accuracy: {acc_test:.4f}')
        print('\n')
        train_losses.append(train_loss_batch_mean)
        train_accuracies.append(train_accuracy_batch_mean)

        model_name = f'{folder}/{folder}_e_{epoch}'
        if save_model:             
            torch.save(model.state_dict(), f"saved_models/{model_name}.pt")
        for sample_id, d in enumerate(data_list):
            words_test = d['words']
            labels_test = true_all[sample_id][:len(words_test)].tolist()
            pred_test = pred_all[sample_id, :, :].max(dim=0)[1][:len(words_test)].tolist()
            data_test_dict = {'words': words_test, 'labels': labels_test, 'pred': pred_test}
            with open(f"saved_models/{model_name}_test_{sample_id}.json", 'w') as f_test:
                json.dump(data_test_dict, f_test)

    if save_results:
        train_losses_np = np.array(train_losses)
        train_accuracies_np = np.array(train_accuracies)
        if test_json != None:
            test_accuracies_np = np.array(test_accuracies)
            data = np.vstack((train_losses_np, train_accuracies_np, test_accuracies_np)).T
            data_df = pd.DataFrame(data, columns=['train_loss', 'train_accuracy', 'test_accuracy'])
            
        else:
            data = np.vstack((train_losses_np, train_accuracies_np)).T
            data_df = pd.DataFrame(data, columns=['train_loss', 'train_accuracy'])
        data_df.to_csv(f'saved_models/{folder}/results.csv', index=False)
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(4, 4))
        fig.tight_layout()

        ax1.plot(range(1, n_epochs + 1), train_losses, 'o-', c='red', label='training')
        ax1.legend(fontsize=12)
        ax1.set_ylabel("loss", fontsize=12)

        ax2.plot(range(1, n_epochs + 1), train_accuracies, 'o-', c='blue', label='training')
        if test_json != None:
            ax2.plot(range(1, n_epochs + 1), test_accuracies, 'o-', c='green', label='test')
        ax2.legend(fontsize=12)
        ax2.set_xlabel("epoch", fontsize=12)
        ax2.set_ylabel("accuracy", fontsize=12)

        plt.show()

    if test_json != None:
        return model, train_losses, train_accuracies, test_accuracies, pred_classes, true_classes, pred_all, true_all, data_list
    else:
        return model, train_losses, train_accuracies

def testing(model, json_file, classes, tokenizer, max_length):
    data_test, target_test, att_mask_test, data_list = ut.preprocess(json_file, classes, tokenizer, 
                                                          n_data=0, batch_size=0, max_length=max_length, test=True)
    with torch.no_grad():
        y_pred_test = model(data_test, attention_mask=att_mask_test)
        y_pred_test = torch.swapaxes(y_pred_test, 1, 2)
        acc_test, predicted_classes, true_classes = ut.accuracy(0, len(classes), y_pred_test, target_test)
        # print(len(predicted_classes), len(true_classes))
    return acc_test, predicted_classes, true_classes, y_pred_test, target_test, data_list