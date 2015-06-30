import sys
import cypico
import numpy as np
from scipy import misc


if __name__ == "__main__":
    try:
        lena = misc.lena().astype(np.uint8)
        results = cypico.detect_frontal_faces(lena)
        print('Found {} faces'.format(len(results)))
        assert(len(results) > 0)
    except:
        sys.exit(1)
