from gpu_extension import (activate_gpu, deactivate_gpu, gpuify, __doc__, __docformat__)

from mdp.utils import fixup_namespace

__all__ = ['activate_gpu', 'deactivate_gpu', 'gpuify']

fixup_namespace(__name__,__all__,('gpu_extension','fixup_namespace',))
