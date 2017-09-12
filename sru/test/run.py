import sys, os
import numpy as np
sys.path.append(os.path.join(".."))
from sru import SRU

def main():
	data = np.random.normal(0, 1, size=(48, 128, 50)).astype(np.float32)
	layer = SRU(128, 128)
	h = layer(data)

if __name__ == "__main__":
	main()
