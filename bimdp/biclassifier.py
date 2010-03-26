
import mdp
import binode


class BiClassifier(binode.BiNode, mdp.ClassifierNode):
    """BiMDP version of the ClassifierNode base class.
    
    It enables that the classification results are returned by execute in the
    message.
    """
    
    def _execute(self, x, return_classify=None, return_prob=None,
                 return_rank=None):
        """Return the unaltered x and classification results when requested.
        
        return_classify -- If True then the 'classify' is called on the x and
            the result is returned in the dict, with the key 'classes'. If
            return_classify is a string then it is used as a prefix for the
            key.
        return_prob, return_rank -- Work like return_classify, but the results
            are stored under the key 'prob' and 'rank'.
        """
        msg = {}
        if return_classify:
            if not isinstance(return_classify, str):
                msg["cl"] = self.classify(x)
            else:
                msg[return_classify + "cl"] = self.classify(x)
        if return_prob:
            if not isinstance(return_prob, str):
                msg["prob"] = self.prob(x)
            else:
                msg[return_prob + "prob"] = self.prob(x)
        if return_rank:
            if not isinstance(return_rank, str):
                msg["rank"] = self.rank(x)
            else:
                msg[return_rank + "rank"] = self.rank(x)
        if msg:
            return x, msg
        else:
            return x
                
                
