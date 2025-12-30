### In this file is defined a Python class to manipulate the simualtion project.
###  - This class must be inherited from th class 'SimulationProject' (no need to import it)
###  - You can use the function "write(input_file, uint, nb_bits=16)"
###            to write an integer of 'nb_bits' bits in the 'input_file'.
### To get this simulation class in Python scripts, please use the functions in manage.py as
###  - search_simulations(repository)
###  - get_simulation(repository='.', classname=None)

def encode(slst):
    mask = 2+8*8380417
    buf = []
    for s in slst:
        s += mask
        for i in range(8):
            buf.append(s % 256)
            s = s // 256
    return bytearray(buf)

def decode(buf):
    slst = []
    mask = 2+8*8380417
    buf = list(buf)
    for j in range(len(buf)//8):
        n = 0
        for i in range(8):
            n = n * 256 + buf[j*8+7-i]
        slst.append(n - mask)
    return slst

class mldsa(SimulationProject):

    @classmethod
    def get_binary_path(cl):
        return 'project.bin'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_input(self, input):
        """ Write into the 'input' file of ELMO tool
                the parameters and the challenges for the simulation """
        super().set_input(input)

    def write_int32(self, input, x):
        mask = 2 + 8*8380417
        x += mask
        for i in range(8):
            byte = x & 0xFF          # 取低 8 位
            write(input, byte, 8)    # 写 8 bits
            x >>= 8                  # 右移 8 位


    
    def set_input_for_each_challenge(self, input, challenge):
        """ Write into the 'input' file of ELMO tool
                the 'challenge' for the simulation """
        for i in range(len(challenge)):
            self.write_int32(input, challenge[i])

    def get_random_challenges(self, nb_challenges=5):
        from random import randint
        return [[randint(-1-4*8380417, 1+4*8380417)] for _ in range(nb_challenges)]

    def get_traces(self):
        from decimal import Decimal
        nb_traces = self.get_number_of_traces()

        # Load the power traces
        if self._complete_results is None:
            self._complete_results = []
            for filename in self.get_results_filenames():
                with open(filename, 'r') as _file:
                    nb = [e for e in _file.readlines()]
                    self._complete_results.append(nb)
        return self._complete_results
