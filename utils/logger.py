# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from PIL import Image


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # old tensorboard code
        # summary = tf.Summary(
        #     value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_array = []
        for i, img in enumerate(images):
            # Write the image to a string

            img = np.clip(img, 0, 1)
            img_np = (img * 255).astype(np.uint8)
            img = img.transpose(1, 2, 0)
            # print("img_np: ", img_np.shape)
            # img_pil = Image.fromarray(img_np, 'RGB')
            # print("img_np: ", img_pil.size)
            img_array.append(img_np)

            with self.writer.as_default():
                tf.summary.image(data=[img], name='%s/%d' %
                                 (tag, i), step=step)
                self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        # self.writer.add_summary(summary, step)
        with self.writer.as_default():
            tf.summary.scalar(tag, histo=hist, step=step)
            self.writer.flush()
        # self.writer.flush()
