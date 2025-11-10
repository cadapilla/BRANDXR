import numpy as np

import gc
import logging
import time, datetime
import numpy as np
import os

from helper import *
from brand import BRANDNode
import integrated_2d_loop_task

class SimFeeding(BRANDNode):

    def __init__(self):
        super().__init__()

        self.in_stream = self.parameters['in_stream']
        self.out_stream = self.parameters['out_stream']
        self.task = self.parameters['task']

        if self.task == 'omni_task_2d':
            self.task_runner = integrated_2d_loop_task.Integrated2DLoopTask(self.parameters, self.r)
            self.task_runner = integrated_2d_loop_task.OmniReach2DRunner(self.parameters, self.r)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        

    def run(self):
        # self.task_runner.run(self.r, self.in_stream, self.out_stream, self.parameters)
        self.task_runner.run(self.parameters)
        self.task_runner.run()

    def cleanup(self):
        # kill issac sim 

