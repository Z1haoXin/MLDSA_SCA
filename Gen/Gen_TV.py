import mldsa44, mldsa65, mldsa87
from random import randint
import numpy as np
from tqdm import trange, tqdm
import argparse
import h5py

def hw_int32(value: int) -> int:
	value &= 0xFFFFFFFF
	return bin(value).count("1")


parser = argparse.ArgumentParser(description = 'key')
parser.add_argument('--level', type = int, default = 2)
parser.add_argument('--target', type = str, default = "s1")
parser.add_argument('--nb_cls', type = int, default = 500)
parser.add_argument('--datatype', type = str, default = "uniform")
parser.add_argument('--filename', type = str, default = "data.h5")
args = parser.parse_args()

def Gen_s1(seed):
	
	if LEVEL == 2:
		return mldsa44.sample_s1(seed)
	elif LEVEL == 3:
		return mldsa65.sample_s1(seed)
	elif LEVEL == 5:
		return mldsa87.sample_s1(seed)
	else:
		print("LEVEL ERROR")

def Gen_c(seed):
	if LEVEL == 2:
		return mldsa44.sample_c(seed)
	elif LEVEL == 3:
		return mldsa65.sample_c(seed)
	elif LEVEL == 5:
		return mldsa87.sample_c(seed)
	else:
		print("LEVEL ERROR")

def NTT(a):
	if LEVEL == 2:
		return mldsa44.NTT(a)
	elif LEVEL == 3:
		return mldsa65.NTT(a)
	elif LEVEL == 5:
		return mldsa87.NTT(a)
	else:
		print("LEVEL ERROR")

def reduce(a):
	if LEVEL == 2:
		return mldsa44.montgomery_reduce(a)
	elif LEVEL == 3:
		return mldsa65.montgomery_reduce(a)
	elif LEVEL == 5:
		return mldsa87.montgomery_reduce(a)
	else:
		print("LEVEL ERROR")

CTILDEBYTES = {2:32, 3:48, 5:64}
L = {2:4, 3:5, 5:7}
N = 256
Q = 8380417

def gen_data_s1hat(nb_cls=500, datatype="uniform", filename="data.h5"):
	if datatype == "uniform":
		data_set = np.zeros([33, nb_cls])
		hw_set = np.zeros([33, nb_cls])
		cnt = [0] * 33
		cnt[0] = nb_cls
		cnt[32] = nb_cls
		data_set[32] = np.array([-1] * nb_cls)
		hw_set[32] = np.array([32] * nb_cls)
		t = 2
		while t < 33:
			seed = [randint(0, 255) for _ in range(64)]
			s1 = Gen_s1(seed)
			s1hat = [NTT(vec) for vec in s1]
			for i in range(L[LEVEL]):
				for j in range(N):
					hw = hw_int32(s1hat[i][j])
					if cnt[hw] < nb_cls:
						data_set[hw][cnt[hw]] = s1hat[i][j]
						hw_set[hw][cnt[hw]] = hw
						cnt[hw] += 1
			t = 0
			for hw in range(33):
				if cnt[hw] == nb_cls:
					t += 1
		print(cnt)
		with h5py.File(filename, "w") as f:
			f.create_dataset("data", data=data_set)
			f.create_dataset("hw", data = hw_set)

	elif datatype == "random":
		data_set = np.array([0] * nb_cls)
		hw_set = np.array([0] * nb_cls)
		cnt = 0
		while cnt < nb_cls:
			seed = [randint(0, 255) for _ in range(64)]
			s1 = Gen_s1(seed)
			s1hat = [NTT(vec) for vec in s1]
			for i in range(L[LEVEL]):
				for j in range(N):
					if cnt < nb_cls:
						data_set[cnt] = s1hat[i][j]
						hw_set[cnt] = hw_int32(s1hat[i][j])
						cnt += 1
		print(cnt)
		with h5py.File(filename, "w") as f:
			f.create_dataset("data", data=data_set)
			f.create_dataset("hw", data = hw_set)

	else:
		print("ERROR: invalid datatype")

def gen_data_xhat(nb_cls=500, datatype="uniform", filename="data.h5"):
	if datatype == "uniform":
		data_set_s1 = np.zeros([33, nb_cls])
		data_set_c = np.zeros([33, nb_cls])
		hw_set = np.zeros([33, nb_cls])
		cnt = [0] * 33
		t = 0
		while t < 33:
			seed_s = [randint(0, 255) for _ in range(64)]
			seed_c = [randint(0, 255) for _ in range(CTILDEBYTES[LEVEL])]
			s1 = Gen_s1(seed_s)
			c = Gen_c(seed_c)
			s1hat = [NTT(vec) for vec in s1]
			chat = NTT(c)
			for i in range(L[LEVEL]):
				for j in range(N):
					xhat = reduce(chat[j] * s1hat[i][j])
					hw = hw_int32(xhat)
					if cnt[hw] < nb_cls:
						data_set_s1[hw][cnt[hw]] = s1hat[i][j]
						data_set_c[hw][cnt[hw]] = chat[j]
						hw_set[hw][cnt[hw]] = hw
						cnt[hw] += 1
			t = 0
			for hw in range(33):
				if cnt[hw] == nb_cls:
					t += 1
		print(cnt)
		with h5py.File(filename, "w") as f:
			f.create_dataset("s1_data", data=data_set_s1)
			f.create_dataset("c_data", data=data_set_c)
			f.create_dataset("hw", data = hw_set)

	elif datatype == "random":
		data_set_s1 = np.zeros([33, nb_cls])
		data_set_c = np.zeros([33, nb_cls])
		hw_set = np.array([0] * nb_cls)
		cnt = 0
		while cnt < nb_cls:
			seed_s = [randint(0, 255) for _ in range(64)]
			seed_c = [randint(0, 255) for _ in range(CTILDEBYTES[LEVEL])]
			s1 = Gen_s1(seed_s)
			c = Gen_c(seed_c)
			s1hat = [NTT(vec) for vec in s1]
			chat = NTT(c)
			for i in range(L[LEVEL]):
				for j in range(N):
					xhat = reduce(chat[j] * s1hat[i][j])
					if cnt < nb_cls:
						data_set_s1[cnt] = s1hat[i][j]
						data_set_c[cnt] = chat[j]
						hw_set[cnt] = hw_int32(xhat)
						cnt += 1
		print(cnt)
		with h5py.File(filename, "w") as f:
			f.create_dataset("s1_data", data=data_set_s1)
			f.create_dataset("c_data", data=data_set_c)
			f.create_dataset("hw", data = hw_set)

	else:
		print("ERROR: invalid datatype")

def encode_c(c):
    buf = []
    for i in range(0, 256, 4):
        word = (c[i+3]<<6) | (c[i+2]<<4) | (c[i+1]<<2) | c[i]
        buf.append(word)
    return buf

def gen_data_chat(nb_cls=500, datatype="uniform", filename="data.h5"):
	if datatype == "uniform":
		data_set = np.zeros((256, 33, nb_cls, 64))
		hw_set = np.zeros((256, 33, nb_cls))

		cnt = np.zeros((256, 33), dtype=int)
		t = [0] * 256
		iter = 0
		while 1:
			iter+=1
			seed = [randint(0, 255) for _ in range(CTILDEBYTES[LEVEL])]
			c = Gen_c(seed)
			ccode = encode_c([ic+1 for ic in c])
			chat = NTT(c)
			for idx in range(256):
				hw = hw_int32(chat[idx])
				if cnt[idx][hw] < nb_cls:
					if hw in [0, 1, 31, 32]:
						cnt[idx][hw] = nb_cls
						for k in range(nb_cls):
							data_set[idx][hw][k] = np.array(ccode)
							hw_set[idx][hw][k] = hw
					else:
						data_set[idx][hw][cnt[idx][hw]] = np.array(ccode)
						hw_set[idx][hw][cnt[idx][hw]] = hw
						cnt[idx][hw] += 1

					# with open(f"Challenge/idx{idx}/hw{hw}.txt", "a") as f:
					# 	line = ','.join(map(str, ccode))
					# 	f.write(line + "\n")

				t[idx] = 0
				for hw in range(33):
					if cnt[idx][hw] == nb_cls:
						t[idx] += 1
			if(iter % 5000==0): print(t)
			if all([item == 33 for item in t]):
				break
			# print(cnt)
		print(cnt)
		with h5py.File(filename, "w") as f:
			f.create_dataset("data", data=data_set)
			f.create_dataset("hw", data = hw_set)

	elif datatype == "random":
		data_set = np.zeros((nb_cls, 64))
		hw_set = np.zeros((nb_cls, 256))
		for k in range(nb_cls):
			seed = [randint(0, 255) for _ in range(CTILDEBYTES[LEVEL])]
			c = Gen_c(seed)
			ccode = encode_c([ic+1 for ic in c])
			chat = NTT(c)
			data_set[k] = np.array(ccode)
			for idx in range(256):
				hw_set[k][idx] = hw_int32(chat[idx])
		with h5py.File(filename, "w") as f:
			f.create_dataset("data", data=data_set)
			f.create_dataset("hw", data = hw_set)

	else:
		print("ERROR: invalid datatype")

def gen_data(target = "s1", nb_cls=500, datatype="uniform", filename="data.h5"):
	assert(target in ["s1", "x", "c"])
	assert(datatype in ["uniform", "random"])
	if target == "s1":
		gen_data_s1hat(nb_cls, datatype, filename)
	elif target == "x":
		gen_data_xhat(nb_cls, datatype, filename)
	else:
		gen_data_chat(nb_cls, datatype, filename)

if __name__ == "__main__":
	LEVEL = args.level
	target = args.target
	nb_cls = args.nb_cls
	datatype = args.datatype
	filename = args.filename
	gen_data(target, nb_cls, datatype, filename)
