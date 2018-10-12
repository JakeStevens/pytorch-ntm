"""An NTM's memory implementation."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys

import time

def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-2:], w, w[:2]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c[1:-1]


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M, use_cuda=False, time=False):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        :param use_cuda: Try to use the GPU
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M
        # Make sure cuda can be used
        if use_cuda and torch.cuda.is_available():
          use_cuda = True
        else:
          use_cuda = False
        self.use_cuda = use_cuda
        self.time = time

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

        # Initialize timer variables
        self.similarity_time = 0
        self.key_similarity_time = 0
        self.content_weighting_time = 0
        self.location_interpolation_time = 0
        self.shift_weighting_time = 0
        self.sharpen_weight_time = 0
        self.read_time = 0
        self.write_time = 0

    def __del__(self):
        if(self.time):
            print("Similarity time: " + str(self.similarity_time))
            print("Key Similarity Time: " + str(self.key_similarity_time))
            print("Content Weighting Time: " + str(self.content_weighting_time))
            print("Location Interpolation Time: " + str(self.location_interpolation_time))
            print("Shift Weighting Time: " + str(self.shift_weighting_time))
            print("Sharpen Time: " + str(self.sharpen_weight_time))
            print("Read Time: " + str(self.read_time))
            print("Write Time: " + str(self.write_time))
        print(self.memory.element_size())

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        if(self.time):
            start = time.time()
            rdVec = torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)
            torch.cuda.synchronize()
            end = time.time()
            self.read_time += (end - start)
        else:
            rdVec = torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)
        return rdVec

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        if self.use_cuda:
          self.memory = self.memory.cuda()
        if self.time:
            start = time.time()
            erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
            add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
            self.memory = self.prev_mem * (1 - erase) + add
            torch.cuda.synchronize()
            end = time.time()
            self.write_time += (end - start)
        else:
            erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
            add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
            self.memory = self.prev_mem * (1 - erase) + add
            
    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).

        Returns a softmax weighting over the rows of the memory matrix.

        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        if(self.time):
            # Content focus
            start = time.time()
            wc = self._similarity(k, β)
            torch.cuda.synchronize()
            end = time.time()
            self.similarity_time += (end - start)
    
            # Location focus
            start = time.time()
            wg = self._interpolate(w_prev, wc, g)
            torch.cuda.synchronize()
            end = time.time()
            self.location_interpolation_time += (end - start)
    
            start = time.time()
            ŵ = self._shift(wg, s)
            torch.cuda.synchronize()
            end = time.time()
            self.shift_weighting_time += (end - start)
    
            start = time.time()
            w = self._sharpen(ŵ, γ)
            torch.cuda.synchronize()
            end = time.time()
            self.sharpen_weight_time += (end - start)
        else:
            # Content focus
            wc = self._similarity(k, β)
            # Location focus
            wg = self._interpolate(w_prev, wc, g)
            ŵ = self._shift(wg, s)
            w = self._sharpen(ŵ, γ)
        return w

    def _similarity(self, k, β):
        if(self.time):
            start = time.time()
            k = k.view(self.batch_size, 1, -1)
            w = F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1)
            torch.cuda.synchronize()
            end = time.time()
            self.key_similarity_time += (end - start)

            start = time.time()
            w = F.softmax(β * w, dim=1)
            torch.cuda.synchronize()
            end = time.time()
            self.content_weighting_time += (end - start)
        else:
            k = k.view(self.batch_size, 1, -1)
            w = F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1)
            w = F.softmax(β * w, dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros(wg.size())
        if self.use_cuda:
          result = result.cuda()
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
