
# coding: utf-8

# # Problem 2

# In this Jupyter notebook, we visualize how HMMs work. This visualization corresponds to problem 2 in set 6.
# 
# Assuming your HMM module is complete and saved at the correct location, you can simply run all cells in the notebook without modification.

# In[6]:


import os
import numpy as np
from IPython.display import HTML

from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)


# ## Visualization of the dataset

# We will be using the Constitution as our dataset. First, we visualize the entirety of the Constitution as a wordcloud:

# In[7]:


text = open(os.path.join(os.getcwd(), 'data/constitution.txt')).read()
wordcloud = text_to_wordcloud(text, title='Constitution')


# ## Training an HMM

# Now we train an HMM on our dataset. We use 10 hidden states and train over 100 iterations:

# In[8]:


obs, obs_map = parse_observations(text)
hmm8 = unsupervised_HMM(obs, 10, 100)


# ## Part G: Visualization of the sparsities of A and O

# We can visualize the sparsities of the A and O matrices by treating the matrix entries as intensity values and showing them as images. What patterns do you notice?

# In[9]:


visualize_sparsities(hmm8, O_max_cols=50)


# ## Generating a sample sentence

# As you have already seen, an HMM can be used to generate sample sequences based on the given dataset. Run the cell below to show a sample sentence based on the Constitution.

# In[5]:


print('Sample Sentence:\n====================')
print(sample_sentence(hmm8, obs_map, n_words=25))


# ## Part H: Using varying numbers of hidden states

# Using different numbers of hidden states can lead to different behaviours in the HMMs. Below, we train several HMMs with 1, 2, 4, and 16 hidden states, respectively. What do you notice about their emissions? How do these emissions compare to the emission above?

# In[ ]:


hmm1 = unsupervised_HMM(obs, 1, 100)
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm1, obs_map, n_words=25))


# In[ ]:


hmm2 = unsupervised_HMM(obs, 2, 100)
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm2, obs_map, n_words=25))


# In[ ]:


hmm4 = unsupervised_HMM(obs, 4, 100)
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm4, obs_map, n_words=25))


# In[ ]:


hmm16 = unsupervised_HMM(obs, 16, 100)
print('\nSample Sentence:\n====================')
print(sample_sentence(hmm16, obs_map, n_words=25))


# ## Part I: Visualizing the wordcloud of each state

# Below, we visualize each state as a wordcloud by sampling a large emission from the state:

# In[ ]:


wordclouds = states_to_wordclouds(hmm8, obs_map)


# ## Visualizing the process of an HMM generating an emission

# The visualization below shows how an HMM generates an emission. Each state is shown as a wordcloud on the plot, and transition probabilities between the states are shown as arrows. The darker an arrow, the higher the transition probability.
# 
# At every frame, a transition is taken and an observation is emitted from the new state. A red arrow indicates that the transition was just taken. If a transition stays at the same state, it is represented as an arrowhead on top of that state.
# 
# Use fullscreen for a better view of the process.

# In[ ]:


anim = animate_emission(hmm8, obs_map, M=8)
HTML(anim.to_html5_video())

