from model import SimpleRNN
from data import Dataset
from plot import Plotter
import torch.nn as nn
import torch
from torch.autograd import Variable
import click


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


@click.command()
@click.option('--cuda', type=bool, default=True)
@click.option('--learning_rate', default=0.01, required=True)
@click.option('--weight_decay', default=0.0, help='L2 norm applied to the weights.')
@click.option('--input_size', default=1, help='Number of channels in the input signal.')
@click.option('--examples_per_class', default=2, help='Duration of the recording.')
@click.option('--epochs', default=10, help='Number of the training epochs.')
@click.option('--bptt_size', default=10, help='Sequence size of example chunks in the mini-batch.')
@click.option('--hidden_size', default=2, help='Number of neurons in each RNN layer.')
@click.option('--num_layers', default=1, help='Number of RNN layers.')
@click.option('--initial_state_type', default='Zeros', type=click.Choice(['Zeros', 'Random']))
@click.option('--data_type', default='Zeros', type=click.Choice(['Zeros', 'SameRandom',
                                                                 'SameDistRandom', 'DiffDistRandom']))
def main(cuda, learning_rate, weight_decay, input_size, examples_per_class, epochs, bptt_size,
         hidden_size, num_layers, initial_state_type, data_type):
    data = Dataset(input_size=input_size, examples_per_class=examples_per_class, seq_size=10000,
                   mode=data_type)
    model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                      output_size=2)

    plotter = Plotter()
    plotter.plot_sequences(data.data, 'input.png')

    criterion = nn.CrossEntropyLoss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    losses_per_epoch = {True: [], False: []}
    losses_last_epoch = {True: [], False: []}

    for epoch in range(epochs):
        for update in [True, False]:
            hidden = model.initial_states(mode=initial_state_type, samples=examples_per_class*2)

            if cuda:
                try:
                    hidden = hidden.cuda()
                except AttributeError:
                    hidden = hidden[0].cuda(), hidden[1].cuda()

            losses = []
            for batch, labels in data.generate_minibatches(size=bptt_size):
                batch = Variable(torch.from_numpy(batch))
                labels = Variable(torch.from_numpy(labels))
                if cuda:
                    batch = batch.cuda()
                    labels = labels.cuda()

                hidden = repackage_hidden(hidden)
                outputs, hidden = model(batch, hidden)

                last = True
                if last:
                    training_outputs = outputs[:, -1, :]
                    training_labels = labels[:, -1]
                else:
                    outputs_num = outputs.size()[-1]
                    training_outputs = outputs.view(-1, outputs_num)
                    training_labels = labels.view(-1)

                loss = criterion(training_outputs, training_labels)
                if update:
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                    optimizer.step()

                l = loss.cpu().data.numpy()[0]
                losses.append(l)

                if epoch == epochs - 1:
                    losses_last_epoch[update].append(l)

            l = sum(losses)/len(losses)
            print('Epoch %d, Update: %s, Loss: %g' % (epoch, update, l))
            losses_per_epoch[update].append(l)

        if epoch > 0:
            plotter.plot_losses(losses_per_epoch, 'losses.png')

    plotter.plot_losses(losses_last_epoch, 'losses_last_epoch.png')


if __name__ == '__main__':
    main()
