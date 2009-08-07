import binet

# this solution is ok, but requires an additional dummy node at the
# top, and every intermediate node has to save its input and output
# during the training

#TODO: handle execution checks for up/down passes: is that possible?
#TODO: how to define global parameters?
#TODO: how to define changing learning rates?

class UpDownBiNode(binet.BiNode):
    """A BiNode that supports an up-pass and a down-pass during learning.

    It can be used for any algorithm that needs an signal to be
    back-propagated in the top-down direction during learning. Typical
    examples would be back-propagation neural networks, that need to
    propagate an error signal down the hierarchy, and any
    probabilistic network where learning is done using a wake-sleep
    approach.
    """

    def __init__(self, *args, **kwargs):
        super(UpDownBiNode, self).__init__(*args, **kwargs)
        self._is_bi_training = True
    
    def is_bi_training(self):
        return self._is_bi_training

    def _stop_bi_training(self):
        """This function is called when the global up-down passes on
        the network are over.
        """
        self._is_bi_training = False
        if hasattr(self, '_save_x'):
            del self._save_x
        if hasattr(self, '_save_y'):
            del self._save_y

    # #### methods to override to implement algorithm
    
    def _up_pass(self, msg=None):
        """Implement up-pass here.
        Coupled to execute"""
        pass

    def _down_pass(self, y, top, msg=None):
        """
        Implement down-pass here
        Returns the top-down signal for the node preceding it in the chain.
        """
        return self._save_x
    
    # #### /methods to override to implement algorithm

    def _down_pass_hook(self, msg=None, stop=False, top=False):
        y = msg['y']
        if y is not None:
            self._check_output(y)
        x = self._down_pass(msg['y'], top, msg=msg)
        # continue down-pass
        if not stop:
            msg['y'] = x
            return msg, -1

    def _up_pass_hook(self, msg=None):
        self._up_pass(msg)
        return {'method': 'up_pass_hook'}, 1
       
    # override execute to perform up-pass during learning
    def execute(self, x, msg=None):
        # normal execution, will call _execute
        y = super(UpDownBiNode, self).execute(x)
        # if call is done during training, save data and call up_pass
        if self.is_bi_training():
            self._save_x = x.copy()
            self._save_y = y.copy()
        return y, msg


class TopUpDownBiNode(UpDownBiNode):
    def __init__(self, bottom_id, top_id=None, *args, **kwargs):
        super(TopUpDownBiNode, self).__init__(*args, **kwargs)
        self._bottom_id = bottom_id
        self._top_id = top_id
        
    def _train(self, x):
        # start up-pass from the bottom
        self._save_x = x
        return {'method': 'up_pass_hook'}, self._bottom_id
    
    def _up_pass_hook(self):
        # start down-pass from the top
        msg = {'method': 'down_pass_hook',
                'y': None,
                self._bottom_id + '=>stop': True}
        if self._top_id is not None:
            msg.update({self._top_id + '=>top': True})
        return msg, 0
    
    def _down_pass(self, y, top, msg=None):
        """
        Generate top activity for down pass.
        Default behavior: return output of the network.
        """
        return self._save_x
    
    def _stop_training(self):
        # stop training
        self._stop_bi_training()
        # send message to everybody that training is over
        return {'method': 'stop_bi_training', '@stop_updown':None}
