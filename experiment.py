import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import copy

from caption_utils import *
from constants import ROOT_STATS_DIR, device
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import nltk

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('using device {}'.format(device))

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__lr = config_data['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_loss = None

        # Init Model
        self.__load_from = config_data['model']['load_from']
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.__model.parameters()),
                                            lr=self.__lr)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, '{}.pt'.format(self.__load_from)))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
            
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        self.__model = self.__model.to(device).float()
        self.__criterion = self.__criterion.to(device)

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model(self.__model)
            
        self.__save_model(self.__best_model, model_name='best_model')

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = []
        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):
            # print(images.size(), captions.size())
            images = images.to(device)
            captions = captions.to(device)
            
            self.__optimizer.zero_grad()
            scores = self.__model(images, captions)
#             print(scores.is_cuda, captions.is_cuda, next(self.__model.parameters()).is_cuda)
            # print(images.size(), scores.size(), captions.size())
            loss = self.__criterion(scores.transpose(1,2), captions)
            training_loss.append(loss.item())
            loss.backward()
            self.__optimizer.step()
            if i % 100 == 0:
                print("epoch {}, iter {}, loss: {}".format(self.__current_epoch+1, i, loss.item()))
#             break
        return np.mean(training_loss)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        ls = []
        gen_caption = []
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                images = images.to(device)
                captions = captions.to(device)
            
                scores = self.__model(images, captions)
                ls.append(self.__criterion(scores.transpose(1,2), captions).item())
                if not gen_caption:
                    gen = self.__model.generate(images, 
                                                    self.__generation_config['max_length'],
                                                    self.__generation_config['temperature']).tolist()
                    gen_captions = self.__vocab.decode(gen)
#                 break
        cur_loss = np.mean(ls)
        if not self.__best_model or cur_loss < self.__best_loss:
            self.__best_model, self.__best_loss = copy.deepcopy(self.__model), cur_loss

        print('epoch {}, sample caption: {}'.format(self.__current_epoch+1, ' '.join(gen_captions[0])))
        return cur_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = []
        b1, b4 = 0, 0
        b1s = []
        b4s = []
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(device)
                captions = captions.to(device)
                # get loss from teacher forcing
                scores = self.__model(images, captions)
                test_loss.append(self.__criterion(scores.transpose(1,2), captions).item())
                
                # auto-regressive generation
                gen = self.__model.generate(images, self.__generation_config['max_length'], self.__generation_config['temperature']).tolist()
                gen_captions = self.__vocab.decode(gen)

                # get BLEU
                for b in range(images.size(0)):
                    # print(img_ids[b])
                    ground_truth = self.__coco_test.imgToAnns[img_ids[b]]
                    refs = [dict['caption'] for dict in ground_truth]
                    ref_tokens = [nltk.tokenize.word_tokenize(s.lower()) for s in refs]
                    # print(refs)
                    # print(gen_captions[b])
                    b1s.append(bleu1(ref_tokens, gen_captions[b]))
                    b4s.append(bleu4(ref_tokens, gen_captions[b]))
                                    
                    if iter % 50 == 0 and b == 0:
                        print(ground_truth)
                        print(gen_captions[b])
                        print(b1s[-1], b4s[-1])
                        print('-' * 20)
                
#                     break
#                 break
        l, b1, b4 = np.mean(test_loss), np.mean(b1s), np.mean(b4s)
        # PPL: https://huggingface.co/docs/transformers/perplexity
        result_str = "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}".format(l,
                                                                                               np.exp(l),
                                                                                               b1,
                                                                                               b4)
        self.__log(result_str)

        return test_loss, b1, b4

    def __save_model(self, model, model_name='latest_model'):
        root_model_path = os.path.join(self.__experiment_dir, '{}.pt'.format(model_name))
        model_dict = model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
