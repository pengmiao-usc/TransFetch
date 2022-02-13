from abc import ABC, abstractmethod

class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass

class NextLineModel(MLPrefetchModel):

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for NextLineModel')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for NextLineModel')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training NextLineModel')

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for NextLineModel')
        prefetches = []
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # Prefetch the next two blocks
            prefetches.append((instr_id, ((load_addr >> 6) + 1) << 6))
            prefetches.append((instr_id, ((load_addr >> 6) + 2) << 6))

        return prefetches

'''
# Example PyTorch Model
import torch
import torch.nn as nn

class PytorchMLModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Initialize your neural network here
        # For example
        self.embedding = nn.Embedding(...)
        self.fc = nn.Linear(...)

    def forward(self, x):
        # Forward pass for your model here
        # For example
        return self.relu(self.fc(self.embedding(x)))

class TerribleMLModel(MLPrefetchModel):
    """
    This class effectively functions as a wrapper around the above custom
    pytorch nn.Module. You can approach this in another way so long as the the
    load/save/train/generate functions behave as described above.

    Disclaimer: It's terrible since the below criterion assumes a gold Y label
    for the prefetches, which we don't really have. In any case, the below
    structure more or less shows how one would use a ML framework with this
    script. Happy coding / researching! :)
    """

    def __init__(self):
        self.model = PytorchMLModel()
    
    def load(self, path):
        self.model = torch.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, data):
        # Just standard run-time here
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = nn.optim.Adam(self.model.parameters())
        scheduler = nn.optim.lr_scheduler.StepLR(optimizer, step_size=0.1)
        for epoch in range(20):
            # Assuming batch(...) is a generator over the data
            for i, (x, y) in enumerate(batch(data)):
                y_pred = self.model(x)
                loss = criterion(y_pred, y)

                if i % 100 == 0:
                    print('Loss:', loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    def generate(self, data):
        self.model.eval()
        prefetches = []
        for i, (x, _) in enumerate(batch(data, random=False)):
            y_pred = self.model(x)
            
            for xi, yi in zip(x, y_pred):
                # Where instr_id is a function that extracts the unique instr_id
                prefetches.append((instr_id(xi), yi))

        return prefetches
'''

# Replace this if you create your own model
Model = NextLineModel
