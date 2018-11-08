import tensorflow as tf
import numpy as np
import scipy.misc 

class Logger(object):
    
    def __init__(self, log_dir ):
        """Create a summary writer logging to log_dir."""
        
        self.train_writer = tf.summary.FileWriter(log_dir + "/train")
        self.test_writer = tf.summary.FileWriter(log_dir + "/eval")

        self.loss = tf.Variable(0.0)
        tf.summary.scalar("loss", self.loss)

        self.merged = tf.summary.merge_all()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def scalar_summary(self, train_loss, test_loss, step):
        """Log a scalar variable."""

        summary = self.session.run(self.merged, {self.loss: train_loss})
        self.train_writer.add_summary(summary, step) 
        self.train_writer.flush()

        summary = self.session.run(self.merged, {self.loss: test_loss})
        self.test_writer.add_summary(summary, step) 
        self.test_writer.flush()

        
        
