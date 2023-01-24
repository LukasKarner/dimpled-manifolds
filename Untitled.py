#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
from models import MnistAEC
from torch.utils.data import DataLoader
from itertools import product


# In[2]:


log_to_stdout()

batch_size = 16
loss_fn = nn.MSELoss()
lr = 0.001
lam = 1.
epochs = 10

device = torch.device('cpu')


# In[3]:


X = torch.linspace(-1, 1, 101)
Y = torch.linspace(-1, 1, 101)
data_2d = torch.tensor(list(product(X, Y)), dtype=X.dtype)
data_2d.unsqueeze_(-1)

M = torch.randn(100,2)
in_place_qr(M)
data = torch.matmul(M, data_2d)
data.squeeze_()
data.unsqueeze_(0)

dataset = torch.utils.data.TensorDataset(data, torch.empty(len(data)))
dataloader = DataLoader(dataset, batch_size, shuffle=True)


# In[4]:


data_2d.size()


# In[5]:


data.size()


# In[6]:

'''
fig, ax = plt.subplots(figsize=(8,8))
plt.plot(data_2d[:,0], data_2d[:,1], ls='', marker='.')
plt.tight_layout()
plt.show()
'''

# In[7]:


class SimpleAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 7),
            nn.ReLU(),
            nn.Linear(7, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 7),
            nn.ReLU(),
            nn.Linear(7, 20),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
        )
    
    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return d


# In[8]:


model = SimpleAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[ ]:

test_iso_ae(dataloader, model, loss_fn, lam, device)
for i in range(epochs):
    logging.info(f'epoch {i + 1}')
    train_iso_ae(dataloader, model, loss_fn, lam, optimizer, device)
    if (i + 1) % 5 == 0:
        test_iso_ae(dataloader, model, loss_fn, lam, device)


# In[ ]:





# In[ ]:





# In[ ]:


quit()


# In[2]:


device = torch.device('cpu')
model = MnistAEC().to(device)
loss = nn.MSELoss().to(device)
iso_loss = IsoLoss(100).to(device)


# In[3]:


x = torch.randn(20,1,28,28).to(device)


# In[4]:


model.train()
y = model.encoder(x)
z = model.decoder(y)
l = loss(x,z)


# In[5]:


l.backward(retain_graph=True)


# In[6]:


model.decoder._modules['0'].weight.grad


# In[7]:


y = torch.unsqueeze(y,1)


# In[8]:


model.eval()
decoder_jac = ft.vmap(ft.jacfwd(model.decoder))(y)


# In[9]:


l_iso = iso_loss(decoder_jac)


# In[10]:


l_iso.backward()


# In[11]:


model.decoder._modules['0'].weight.grad


# In[ ]:




