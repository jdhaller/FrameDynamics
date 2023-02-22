
"""
Date: 2022/12
Author: Jens D. Haller
Mail: jens.haller@kit.edu / jhaller@gmx.de
Institution: Karlsruhe Institute of Technology

"""
    
import re
import numpy as np

class ShapeLoader():

    @classmethod
    def load_amp_phase(cls, path_to_file: str, separator: str, \
                       start: str, end: str) -> list:

        data = []

        with open(path_to_file, 'r') as tmp:
            load_data = tmp.read()
        load_data = load_data.split('\n')

        flag = False
        for line in load_data:
            if end in line:
                flag = False
            if flag:

                # substitute whitespaces at the beginning
                line = re.sub("^\s*", "", line)

                # split by separator (flanked by whitespaces)
                line = re.split("\s*"+separator+"\s*", line)

                try:
                    line = [float(slic) for slic in line]
                except:
                    line = [slic.replace(' ', '') for slic in line]

                data.append(line)
            if start in line:
                flag = True

        return data
    # ====================================================================


    @classmethod
    def load_XY_amp(cls, path_to_file: str, separator: str, \
                    comment: str) -> list:

        data = []

        with open(path_to_file, 'r') as tmp:
            load_data = tmp.read()
        load_data = load_data.split('\n')

        for line in load_data:
        
            # don't use comments, marked by first char 
            # (not counting whitespaces) 
            if not re.search(r"\s*"+comment, line):
                
                # substitute whitespaces at the beginning
                line = re.sub("^\s*", "", line)

                # split by separator (flanked by whitespaces)
                line = re.split("\s*"+separator+"\s*", line)

                try:
                    line = [float(slic) for slic in line]
                    data.append(line)

                except ValueError:
                    if line == ['']: 
                        pass
                    else: 
                        raise TypeError("Unable to convert to float: "+\
                                        "Check separator!")
            
        return cls.convert2Bruker( np.array(data) )
    # ====================================================================

    @classmethod
    def convert2Bruker(data):

        amplitudes = np.sqrt(data[:,0]**2 + data[:,1]**2)
        max_amp = max(amplitudes)
        amplitudes = amplitudes / max_amp * 100

        length = np.sum(data[:,-1])

        phases = np.arctan2(data[:,0], data[:,1])/np.pi * 360
        shape = np.vstack([phases, amplitudes]).T

        return max_amp, length, shape
    # ====================================================================

