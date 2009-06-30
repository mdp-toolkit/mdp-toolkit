import os
import re
import mdp

def get_svn_revision():
    revision = ''
    # if we are here, mdp has not been installed
    # we are running from a working copy
    wc = os.path.dirname(mdp.__file__)
    # svn entries file
    entries = os.path.join(wc, '.svn', 'entries')

    # try open it, if it doesn't work something is broken in the
    # user svn system: we'll return an empty string in this case
    try:
        fl = open(entries, 'r')
        text = fl.read()
        m = re.search(r'dir[\n\r]+(?P<revision>\d+)', text)
        if m:
            revision = m.group('revision')
    except:
        pass
    return revision
