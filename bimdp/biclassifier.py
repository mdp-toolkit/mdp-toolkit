
import mdp
from . import binode


class BiClassifier(binode.BiNode, mdp.ClassifierNode):
    """BiMDP version of the ClassifierNode base class.

    It enables that the classification results are returned by execute in the
    message.
    """

    def _execute(self, x, return_labels=None, return_probs=None,
                 return_ranks=None):
        """Return the unaltered x and classification results when requested.

        return_labels -- If True then the 'label' method is called on the x
            and the result is returned in the dict, with the key 'labels'. If
            return_labels is a string then this is used as a prefix for the
            'labels' key of the result.
        return_probs, return_ranks -- Work like return_labels, but the results
            are stored under the key 'probs' and 'ranks'.
        """
        msg = {}
        if return_labels:
            if not isinstance(return_labels, str):
                msg["labels"] = self.label(x)
            else:
                msg[return_labels + "labels"] = self.label(x)
        if return_probs:
            if not isinstance(return_probs, str):
                msg["probs"] = self.prob(x)
            else:
                msg[return_probs + "probs"] = self.prob(x)
        if return_ranks:
            if not isinstance(return_ranks, str):
                msg["ranks"] = self.rank(x)
            else:
                msg[return_ranks + "ranks"] = self.rank(x)
        if msg:
            return x, msg
        else:
            return x
