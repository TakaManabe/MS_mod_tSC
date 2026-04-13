import numpy as np
import subprocess
import sys
import scipy.io
import os
try:
    import mat73
except ImportError:
    print(f"\033[1;33m -- mat73 is not installed. Installing now...\033[0m")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'mat73'])
    import mat73
    print(f"\033[1;33m -- installed.\033[0m")

######################################################################
def matFileLoad(DataDir, DataFolder):

    # File path definition
    loadfilepath = os.path.join(DataDir, DataFolder)
    loadfilename = os.path.join(loadfilepath)

    # File load
    try:
        mat = mat73.loadmat(loadfilename)
    except Exception as e:
        mat = matDictLoad(loadfilename)

    return mat

######################################################################
def matDictLoad(filename):
    '''
    この関数は、scipy.io.loadmatの代わりに呼び出すことで、
    MATLABの構造体を適切にPythonの辞書に変換します。
    '''
    def _check_keys(dict):
        '''
        辞書内のエントリがmat_structの場合、再帰的に辞書に変換します。
        '''
        for key in dict:
            if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
                dict[key] = _todict(dict[key])
            elif isinstance(dict[key], np.ndarray) and dict[key].dtype == object:  # np.objectをobjectに置き換え
                dict[key] = _tolist(dict[key])
        return dict

    def _todict(matobj):
        '''
        mat_structオブジェクトから再帰的に辞書を構築します。
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = getattr(matobj, strg)
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray) and elem.dtype == object:  # np.objectをobjectに置き換え
                dict[strg] = _tolist(elem)
            else:
                dict[strg] = elem
        return dict

    def _tolist(ndarray):
        '''
        オブジェクト型のnumpy配列をリストに変換し、内容を再帰的に処理します。
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray) and sub_elem.dtype == object:  # np.objectをobjectに置き換え
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

