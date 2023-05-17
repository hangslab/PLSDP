import os
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras import optimizers
from keras.layers import *
from keras.models import Model
from keras import backend as K
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio import SeqIO
import sys
from hier_attention_mask import Attention

def convertSampleToPhysicsVector_pca(seq):
    """
    Convertd the raw data to physico-chemical property
    PARAMETER
    seq: "MLHRPVVKEGEWVQAGDLLSDCASSIGGEFSIGQ" one fasta seq
        X denoted the unknow amino acid.
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    将原始数据转换为理化性质
    参数
    序列：“MLHRPVVKEGEWVQAGDLLSDCASSIGGEFSIGQ”一个 fasta 序列
         X 表示未知氨基酸。
    probMatr：样本的概率矩阵。
         形状 (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """
    letterDict = {}
    letterDict["A"] = [0.008, 0.134, -0.475, -0.039, 0.181]
    letterDict["R"] = [0.171, -0.361, 0.107, -0.258, -0.364]
    letterDict["N"] = [0.255, 0.038, 0.117, 0.118, -0.055]
    letterDict["D"] = [0.303, -0.057, -0.014, 0.225, 0.156]
    letterDict["C"] = [-0.132, 0.174, 0.070, 0.565, -0.374]
    letterDict["Q"] = [0.149, -0.184, -0.030, 0.035, -0.112]
    letterDict["E"] = [0.221, -0.280, -0.315, 0.157, 0.303]
    letterDict["G"] = [0.218, 0.562, -0.024, 0.018, 0.106]
    letterDict["H"] = [0.023, -0.177, 0.041, 0.280, -0.021]
    letterDict["I"] = [-0.353, 0.071, -0.088, -0.195, -0.107]
    letterDict["L"] = [-0.267, 0.018, -0.265, -0.274, 0.206]
    letterDict["K"] = [0.243, -0.339, -0.044, -0.325, -0.027]
    letterDict["M"] = [-0.239, -0.141, -0.155, 0.321, 0.077]
    letterDict["F"] = [-0.329, -0.023, 0.072, -0.002, 0.208]
    letterDict["P"] = [0.173, 0.286, 0.407, -0.215, 0.384]
    letterDict["S"] = [0.199, 0.238, -0.015, -0.068, -0.196]
    letterDict["T"] = [0.068, 0.147, -0.015, -0.132, -0.274]
    letterDict["W"] = [-0.296, -0.186, 0.389, 0.083, 0.297]
    letterDict["Y"] = [-0.141, -0.057, 0.425, -0.096, -0.091]
    letterDict["V"] = [-0.274, 0.136, -0.187, -0.196, -0.299]
    letterDict["X"] = [0, -0.00005, 0.00005, 0.0001, -0.0001]
    letterDict["-"] = [0, 0, 0, 0, 0, 1]
    AACategoryLen = 5  # 6 for '-'
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr


def convertSampleToBlosum62(seq):
    letterDict = {}
    letterDict["A"] = [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0]
    letterDict["R"] = [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3]
    letterDict["N"] = [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3]
    letterDict["D"] = [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3]
    letterDict["C"] = [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1]
    letterDict["Q"] = [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2]
    letterDict["E"] = [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2]
    letterDict["G"] = [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3]
    letterDict["H"] = [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3]
    letterDict["I"] = [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3]
    letterDict["L"] = [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1]
    letterDict["K"] = [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2]
    letterDict["M"] = [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1]
    letterDict["F"] = [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1]
    letterDict["P"] = [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2]
    letterDict["S"] = [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2]
    letterDict["T"] = [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0]
    letterDict["W"] = [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3]
    letterDict["Y"] = [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1]
    letterDict["V"] = [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]
    AACategoryLen = 20  # 6 for '-'
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr


def convertSampleToCBOW(seq):
    letterDict = {}
    letterDict["A"] = [3.999999999999999, 1.9224027401898192, -2.54542767330058, 0.9054695577496974, -0.08997286113781003, 0.5461237572712727, 0.05793958592865156, 3.3671880224530577, 0.44662607721985964, -0.8619949581416184, 1.774648418695889, -0.996623090551424, 1.0414558763035118, -1.3360414641419274, 1.6835945877273706, -0.925293605180982, 1.9055765222710177, 2.014607058856954, -3.0, 2.423100449094865  ]
    letterDict["R"] = [1.9224027401898192, 5.000000110498654, -3.1826467872500617, 1.83162457556425, 1.7148277908877556, 0.7777577753344023, 1.3667850146804825, 0.7716617541186306, 3.60204094167071, -2.2211551801509657, 0.7433080752115466, -0.12666570386491027, 0.24589742620452126, -2.808030965628724, 0.5813757651977289, -1.1496327991154422, -0.1280971127900834, 4.079494856637261, -0.9984044920773061, 0.2738966547375379  ]
    letterDict["N"] = [-2.54542767330058, -3.1826467872500617, 6.000000121760412, 0.26402496717598956, -3.7032836734184174, 0.07777614680931788, -1.0713996216935233, -4.0, -2.188005032896912, 0.9004257139611458, -3.9999997938393, 1.6439052944030208, -1.831874593574947, -0.8938046355200927, -1.4147921395216754, 0.5974485850718703, 1.2567930196043424, -4.000000081173608, 1.3213180551568375, -2.9999999256350542 ]
    letterDict["D"] = [0.9054695577496974, 1.83162457556425, 0.26402496717598956, 6.000000137684608, -2.410490336438102, -0.7066777080154099, 3.6162240217427426, -0.6870400971840516, -0.7990009432730142, 0.9328218696448383, -0.709728414201586, 2.9958671369295686, -1.4377954860203672, -1.7512797976783625, -1.2560843228146892, -0.4881909332814911, 1.1624214136493156, -2.1829064413287305, 0.3459786113981379, 0.6099861512640505 ]
    letterDict["C"] = [-0.08997286113781003, 1.7148277908877556, -3.7032836734184174, -2.410490336438102, 9.00000018127487, -1.6538057110360866, -3.812542673994483, -0.5853925526355963, 3.4008149351364887, -4.0000000805666085, -3.0699812496843166, -2.901770962613418, -2.9999999255858922, -2.877889883779649, -0.7185234512953114, -2.681012186972212, -1.9999999509486772, 4.060252145519927, -1.1968450046321744, -1.983922769723393  ]
    letterDict["Q"] = [0.5461237572712727, 0.7777577753344023, 0.07777614680931788, -0.7066777080154099, -1.6538057110360866, 5.0, 0.4127267975171167, -2.2119219492892217, 2.370508671048586, -2.457420224958085, -0.37305995050081586, 0.2673891657688095, -0.5505880252420592, -4.000000184637939, 2.2111693779986035, -0.2027219839370118, 0.5473420559665785, 1.142449555675963, -1.9188006516740579, -2.204756094380816 ]
    letterDict["E"] = [0.05793958592865156, 1.3667850146804825, -1.0713996216935233, 3.6162240217427426, -3.812542673994483, 0.4127267975171167, 5.0, -2.789262296858721, -1.8861718601251714, 0.12956858997960863, -0.4631783588989782, 3.973501560615596, -1.7216652905017633, -3.7081476106465927, -1.6132130511185179, -1.319719165016221, -0.16006677847098194, -2.5565970602977854, -1.0525917650532088, -0.518037056449042  ]
    letterDict["G"] = [3.3671880224530577, 0.7716617541186306, -4.0, -0.6870400971840516, -0.5853925526355963, -2.2119219492892217, -2.789262296858721, 6.0, 1.339686512898458, -1.2417086083004554, -1.0537456822218352, -1.8991476931399809, 1.0135718186430955, 0.12202610595817398, 2.313964722436334, -1.1625184262645654, 1.8681417789247068, 4.889648152996016, -0.5004551802020044, 2.092511132680251 ]
    letterDict["H"] = [0.44662607721985964, 3.60204094167071, -2.188005032896912, -0.7990009432730142, 3.4008149351364887, 2.370508671048586, -1.8861718601251714, 1.339686512898458, 7.999999819356918, -3.006373221592818, -0.8958043205097943, -2.1903629145649015, -0.12912517087959663, -0.8983356505394658, 1.8304612858524627, -1.4714381538498924, 0.06860353581555922, 4.808090898206266, 0.8333425780609708, -1.488885433810885 ]
    letterDict["I"] = [-0.8619949581416184, -2.2211551801509657, 0.9004257139611458, 0.9328218696448383, -4.0000000805666085, -2.457420224958085, 0.12956858997960863, -1.2417086083004554, -3.006373221592818, 4.0000000805666085, -0.3042653618068747, 2.458232141732212, 0.8967297628849018, 2.364101214825318, -4.0, -1.8206629324066537, 0.8815856428739957, -2.253704946110871, 2.407370545890224, 1.2902250650123148  ]
    letterDict["L"] = [1.774648418695889, 0.7433080752115466, -3.9999997938393, -0.709728414201586, -3.0699812496843166, -0.37305995050081586, -0.4631783588989782, -1.0537456822218352, -0.8958043205097943, -0.3042653618068747, 3.9999997938393004, 0.1523951601433195, 2.541163280031004, 1.4493037764920822, -1.0058764997021414, -1.3520625782189768, 0.8210821359496656, 2.9776568832175596, -0.5701620491327464, 1.7831403166380067  ]
    letterDict["K"] = [-0.996623090551424, -0.12666570386491027, 1.6439052944030208, 2.9958671369295686, -2.901770962613418, 0.2673891657688095, 3.973501560615596, -1.8991476931399809, -2.1903629145649015, 2.458232141732212, 0.1523951601433195, 5.000000202951467, -2.2804114336185477, -2.0308723981052452, -2.6673707731299494, -1.046172440189062, -0.2423814527628403, -3.998710942811164, 0.7589893844867887, -1.5414268768993424 ]
    letterDict["M"] = [1.0414558763035118, 0.24589742620452126, -1.831874593574947, -1.4377954860203672, -2.9999999255858922, -0.5505880252420592, -1.7216652905017633, 1.0135718186430955, -0.12912517087959663, 0.8967297628849018, 2.541163280031004, -2.2804114336185477, 4.999999875976487, 1.584233258647287, -0.38064192803137864, -2.3352684414527163, 0.5143108092928712, 2.524679080140674, 0.4891764238163504, 1.181682486073677  ]
    letterDict["F"] = [-1.3360414641419274, -2.808030965628724, -0.8938046355200927, -1.7512797976783625, -2.877889883779649, -4.000000184637939, -3.7081476106465927, 0.12202610595817398, -0.8983356505394658, 2.364101214825318, 1.4493037764920822, -2.0308723981052452, 1.584233258647287, 6.000000276956908, -2.2206261116318595, -1.762669615193011, 0.43181059224076757, 4.579805954490016, 4.36858434768749, 0.049760063981616476 ]
    letterDict["P"] = [1.6835945877273706, 0.5813757651977289, -1.4147921395216754, -1.2560843228146892, -0.7185234512953114, 2.2111693779986035, -1.6132130511185179, 2.313964722436334, 1.8304612858524627, -4.0, -1.0058764997021414, -2.6673707731299494, -0.38064192803137864, -2.2206261116318595, 7.0, 1.6227195781339228, 2.7917716651109226, 2.0657560205450403, -1.662031020626987, -1.2075151206244037  ]
    letterDict["S"] = [-0.925293605180982, -1.1496327991154422, 0.5974485850718703, -0.4881909332814911, -2.681012186972212, -0.2027219839370118, -1.319719165016221, -1.1625184262645654, -1.4714381538498924, -1.8206629324066537, -1.3520625782189768, -1.046172440189062, -2.3352684414527163, -1.762669615193011, 1.6227195781339228, 3.9999999027958344, 3.2662186237081725, -1.5262540612170765, -1.784485022635621, -1.889845709593782 ]
    letterDict["T"] = [1.9055765222710177, -0.1280971127900834, 1.2567930196043424, 1.1624214136493156, -1.9999999509486772, 0.5473420559665785, -0.16006677847098194, 1.8681417789247068, 0.06860353581555922, 0.8815856428739957, 0.8210821359496656, -0.2423814527628403, 0.5143108092928712, 0.43181059224076757, 2.7917716651109226, 3.2662186237081725, 4.999999877371693, -0.3352757467464603, 0.18984110464005333, 0.9806717078167575  ]
    letterDict["W"] = [2.014607058856954, 4.079494856637261, -4.000000081173608, -2.1829064413287305, 4.060252145519927, 1.142449555675963, -2.5565970602977854, 4.889648152996016, 4.808090898206266, -2.253704946110871, 2.9776568832175596, -3.998710942811164, 2.524679080140674, 4.579805954490016, 2.0657560205450403, -1.5262540612170765, -0.3352757467464603, 11.000000223227422, 1.652173046811678, -1.160827704650595 ]
    letterDict["Y"] = [-3.0, -0.9984044920773061, 1.3213180551568375, 0.3459786113981379, -1.1968450046321744, -1.9188006516740579, -1.0525917650532088, -0.5004551802020044, 0.8333425780609708, 2.407370545890224, -0.5701620491327464, 0.7589893844867887, 0.4891764238163504, 4.36858434768749, -1.662031020626987, -1.784485022635621, 0.18984110464005333, 1.652173046811678, 7.0, -1.6458009520473016  ]
    letterDict["V"] = [2.423100449094865, 0.2738966547375379, -2.9999999256350542, 0.6099861512640505, -1.983922769723393, -2.204756094380816, -0.518037056449042, 2.092511132680251, -1.488885433810885, 1.2902250650123148, 1.7831403166380067, -1.5414268768993424, 1.181682486073677, 0.049760063981616476, -1.2075151206244037, -1.889845709593782, 0.9806717078167575, -1.160827704650595, -1.6458009520473016, 3.9999999008467393  ]
    AACategoryLen = 20  # 6 for '-'
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr


def convertSampleToSG(seq):
    letterDict = {}
    letterDict["A"] = [4.0, 0.9690403032189927, -3.432569348305389, 0.6249732786643705, -3.9999999999999982, 0.23123236787720813, -1.5847239816476595, 3.5470851788235187, -2.8139191589117, -2.995226521982982, 0.46073657914112687, -2.9999999999999982, 1.6237942213506447, -3.3639522913105147, 2.568318977878901, -1.627354662319222, 1.5094598357794524, 4.387751915739621, -3.0, 2.5439113802356417  ]
    letterDict["R"] = [0.9690403032189927, 5.000000000000002, -2.5108877836458205, 2.1998852892421006, 0.9918473460413715, 0.31294234756867745, 1.832916611381524, 0.5777759811076457, 2.6141947858208425, -2.9581289758777647, 0.9369515259211916, -0.4727754930179735, 0.23077705182948982, -1.769872992565439, 1.7384239431772581, -0.45249421474642126, 0.41035694383831967, 5.454971490238158, 0.4920802979431045, 1.0162286804137572  ]
    letterDict["N"] = [-3.432569348305389, -2.5108877836458205, 6.0, 0.9380084942293188, -3.8377195973125655, -1.3974416766197582, -2.2812279586862854, -3.9999999999999982, -3.0, 0.642322252072951, -2.937717311430717, 0.702039645765069, -1.4107887812352207, -1.7292102352034853, -3.9999999999999982, -1.4481976142712671, 0.28973670931359763, -2.119832074846393, 2.2095591924335096, -1.144587401234567  ]
    letterDict["D"] = [0.6249732786643705, 2.1998852892421006, 0.9380084942293188, 6.0, -2.009861048756145, 0.46410417081120237, 3.028629370727039, 1.0359360945968987, -0.11795695105491255, 0.4541041442939502, 0.2746924442736294, 1.664535665645115, 1.1104211367867407, -0.6866841032985533, 0.5986353555245412, -0.5370313299775837, 2.194526360625355, 3.7563646864215343, 3.178287286991644, 2.167354217494516]
    letterDict["C"] = [ -3.9999999999999982, 0.9918473460413715, -3.8377195973125655, -2.009861048756145, 9.0, -2.9999999999999982, -4.0, -3.2155845874683067, 3.000552674575548, -4.0, -4.0, -0.3626889629601333, -2.9999999999999982, -2.0856294255983308, -3.829093096954285, -3.0000000000000018, -2.0, -4.0, -0.37378668955275884, -3.0  ]
    letterDict["Q"] = [0.23123236787720813, 0.31294234756867745, -1.3974416766197582, 0.46410417081120237, -2.9999999999999982, 5.000000000000002, -0.5793048737054551, -1.9583978877066066, 0.4574937841869726, -3.394549211839349, -0.9821285526437968, -0.2270644751459283, 0.25105273546498097, -4.0, 0.37301290078963056, -2.089669021681045, 0.0007092035234634864, 1.2457238971328337, -0.46016873053569185, -0.4843684701300237 ]
    letterDict["E"] = [-1.5847239816476595, 1.832916611381524, -2.2812279586862854, 3.028629370727039, -4.0, -0.5793048737054551, 5.0, -1.4635755510789519, -0.37378987000071007, -1.7163905911012591, -1.0555158508244062, 2.8072597874463927, 0.20147228182382548, -3.8965214606483407, -0.4896278655144055, -1.6403128002228282, 0.5322020574978303, -0.11790359555134344, 0.5161708730555574, 0.8228573861003277  ]
    letterDict["G"] = [3.5470851788235187, 0.5777759811076457, -3.9999999999999982, 1.0359360945968987, -3.2155845874683067, -1.9583978877066066, -1.4635755510789519, 6.0000000000000036, -0.508162250479506, -0.2049976873396986, 1.2025797759213717, -1.2492580375314652, 1.9186481930893091, 0.035218935196830614, 1.75287244517523, -0.6945129662167293, 2.2722077765916175, 6.1297693811988445, 0.18984250293326355, 3.3106068982466716  ]
    letterDict["H"] = [-2.8139191589117, 2.6141947858208425, -3.0, -0.11795695105491255, 3.000552674575548, 0.4574937841869726, -0.37378987000071007, -0.508162250479506, 8.0, -2.5019633365584593, -1.5957818461332494, -0.3165947346388034, -1.5012352796668047, -0.16947910143082012, -1.272369414756323, -1.6487146293872659, -0.654762356996482, 1.145087708582782, 1.0710233413460202, -1.6480782086501016 ]
    letterDict["I"] = [-2.995226521982982, -2.9581289758777647, 0.642322252072951, 0.4541041442939502, -4.0, -3.394549211839349, -1.7163905911012591, -0.2049976873396986, -2.5019633365584593, 4.0, 1.546283601178331, 2.6252627973779017, 2.1732245783591058, 3.932079922495415, -0.7761227845439063, -0.3148482418773355, 2.3906713863149136, 4.216653589678444, 5.008607548011195, 2.0217335490446526 ]
    letterDict["L"] = [0.46073657914112687, 0.9369515259211916, -2.937717311430717, 0.2746924442736294, -4.0, -0.9821285526437968, -1.0555158508244062, 1.2025797759213717, -1.5957818461332494, 1.546283601178331, 4.0, 0.9923961161326655, 3.284497446085206, 3.6192528704330442, 2.2977907577003247, 0.34737299606242367, 2.3612250825335224, 8.353964495886622, 3.654329483960126, 2.8523414432115324 ]
    letterDict["K"] = [-2.9999999999999982, -0.4727754930179735, 0.702039645765069, 1.664535665645115, -0.3626889629601333, -0.2270644751459283, 2.8072597874463927, -1.2492580375314652, -0.3165947346388034, 2.6252627973779017, 0.9923961161326655, 5.000000000000002, -0.27772818735240534, -0.44925509428232857, -2.6561670697665516, -1.2329771387057082, 0.6241077750063031, -2.183581037150333, 3.0961204579849095, -0.4983544560334696 ]
    letterDict["M"] = [1.6237942213506447, 0.23077705182948982, -1.4107887812352207, 1.1104211367867407, -2.9999999999999982, 0.25105273546498097, 0.20147228182382548, 1.9186481930893091, -1.5012352796668047, 2.1732245783591058, 3.284497446085206, -0.27772818735240534, 5.000000000000002, 2.2518453172889963, 0.4926086935739846, -1.2012386246075675, 1.7779481075991264, 5.063861721962873, 2.197123607472083, 2.509666985164804  ]
    letterDict["F"] = [-3.3639522913105147, -1.769872992565439, -1.7292102352034853, -0.6866841032985533, -2.0856294255983308, -4.0, -3.8965214606483407, 0.035218935196830614, -0.16947910143082012, 3.932079922495415, 3.6192528704330442, -0.44925509428232857, 2.2518453172889963, 6.0, -0.41118243695390255, -0.17762977964376603, 1.6636208204925413, 6.897192826606364, 5.516263834801158, 1.3295912256217068  ]
    letterDict["P"] = [2.568318977878901, 1.7384239431772581, -3.9999999999999982, 0.5986353555245412, -3.829093096954285, 0.37301290078963056, -0.4896278655144055, 1.75287244517523, -1.272369414756323, -0.7761227845439063, 2.2977907577003247, -2.6561670697665516, 0.4926086935739846, -0.41118243695390255, 7.000000000000002, -0.19418336500380917, 1.7719325900704934, 2.761571508236333, -1.4603235721608083, 0.9056713640010727  ]
    letterDict["S"] = [-1.627354662319222, -0.45249421474642126, -1.4481976142712671, -0.5370313299775837, -3.0000000000000018, -2.089669021681045, -1.6403128002228282, -0.6945129662167293, -1.6487146293872659, -0.3148482418773355, 0.34737299606242367, -1.2329771387057082, -1.2012386246075675, -0.17762977964376603, -0.19418336500380917, 3.9999999999999982, 2.90270541253002, 0.5307065874409602, -0.3990978310960287, 0.41348010127330603  ]
    letterDict["T"] = [1.5094598357794524, 0.41035694383831967, 0.28973670931359763, 2.194526360625355, -2.0, 0.0007092035234634864, 0.5322020574978303, 2.2722077765916175, -0.654762356996482, 2.3906713863149136, 2.3612250825335224, 0.6241077750063031, 1.7779481075991264, 1.6636208204925413, 1.7719325900704934, 2.90270541253002, 5.0, 3.2157164557181623, 1.4086518857964236, 2.538260202359206 ]
    letterDict["W"] = [4.387751915739621, 5.454971490238158, -2.119832074846393, 3.7563646864215343, -4.0, 1.2457238971328337, -0.11790359555134344, 6.1297693811988445, 1.145087708582782, 4.216653589678444, 8.353964495886622, -2.183581037150333, 5.063861721962873, 6.897192826606364, 2.761571508236333, 0.5307065874409602, 3.2157164557181623, 11.0, 2.776462625333366, 1.8477717222378534 ]
    letterDict["Y"] = [-3.0, 0.4920802979431045, 2.2095591924335096, 3.178287286991644, -0.37378668955275884, -0.46016873053569185, 0.5161708730555574, 0.18984250293326355, 1.0710233413460202, 5.008607548011195, 3.654329483960126, 3.0961204579849095, 2.197123607472083, 5.516263834801158, -1.4603235721608083, -0.3990978310960287, 1.4086518857964236, 2.776462625333366, 7.0, 0.640653516414881  ]
    letterDict["V"] = [2.5439113802356417, 1.0162286804137572, -1.144587401234567, 2.167354217494516, -3.0, -0.4843684701300237, 0.8228573861003277, 3.3106068982466716, -1.6480782086501016, 2.0217335490446526, 2.8523414432115324, -0.4983544560334696, 2.509666985164804, 1.3295912256217068, 0.9056713640010727, 0.41348010127330603, 2.538260202359206, 1.8477717222378534, 0.640653516414881, 4.0  ]
    AACategoryLen = 20  # 6 for '-'
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr




def convertlabels_to_categorical(seq):
    label = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0.]])
    for index in seq.split(";"):
        i = int(index.split(".")[0])
        j = int(index.split(".")[1])
        label[i][j] = 1.0
    return label


def readPSSM(pssmfile):
    pssm = []
    with open(pssmfile, 'r') as f:
        count = 0
        for eachline in f:
            count += 1
            if count <= 3:
                continue
            if not len(eachline.strip()):
                break
            line = eachline.split()
            pssm.append(line[2: 22])  # 22:42
    return np.array(pssm)


def getTrue4out1(y):  # input [?, 10, 8]  output [?, 10]  elements are 0 or 1
    if len(y.shape) == 2:
        label = y.sum(axis=1)
        label[label >= 1] = 1
    if len(y.shape) == 3:
        label = y.sum(axis=2)
        label[label >= 1] = 1
    return label


def getTrue4out2(y):  # [?,10,8]
    label = []
    if len(y.shape) == 2:
        x = y
        a = []
        a.extend(x[0][0:8])
        a.extend(x[1][0:8])
        a.extend(x[2][0:2])
        a.extend(x[3][0:5])
        a.extend(x[4][0:6])
        a.extend(x[5][0:5])
        a.extend(x[6][0:5])
        a.extend(x[7][0:4])
        a.extend(x[8][0:1])
        a.extend(x[9][0:1])
        label.append(a)
    else:
        for x in y:
            a = []
            a.extend(x[0][0:8])
            a.extend(x[1][0:8])
            a.extend(x[2][0:2])
            a.extend(x[3][0:5])
            a.extend(x[4][0:6])
            a.extend(x[5][0:5])
            a.extend(x[6][0:5])
            a.extend(x[7][0:4])
            a.extend(x[8][0:1])
            a.extend(x[9][0:1])
            label.append(a)
    return np.array(label)


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], y_true[:, i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis=-1)

    return weighted_loss


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.1 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def gen_mask_mat(num_want, num_mask):
    seq_want = np.ones(num_want)
    seq_mask = np.zeros(num_mask)
    seq = np.concatenate([seq_want, seq_mask])
    return seq

def mask_func(x):
    return x[0] * x[1]

def plus(x):
    return x[0] + x[1]


def singlemodel(train_x):
    '''MULocDeep模型训练参数
    
      LSTM的维数为180，
      权重矩阵的维数是369×180，
      参数矩阵的维数是41×369，
      正则权重为0.00001，
      注意力正则权重为0.0007159，
      学习率为0.0005
      
    '''
    [dim_gru, da, r, W_regularizer, Att_regularizer_weight, drop_per, drop_hid, lr] = [
        180, 369, 41, 0.00001,0.0007159, 0.1, 0.1, 0.0005]
    
    input = Input(shape=(train_x.shape[1:]), name="Input")  # input's shape=[?,seq_len,encoding_dim]
    input_mask = Input(shape=([train_x.shape[1], 1]), dtype='float32')  # (batch_size,max_len,1)
    l_indrop = layers.Dropout(drop_per)(input)
    mask_input = []
    mask_input.append(l_indrop)
    mask_input.append(input_mask)
    mask_layer1 = Lambda(mask_func)(mask_input)
    x1 = layers.Conv1D(filters=25, kernel_size=1, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer1) 
    # [?,seq_len,dim_conv]
    x1bn = layers.BatchNormalization()(x1)
    x1d = layers.Dropout(drop_hid)(x1bn)
    mask_input = []
    mask_input.append(x1d)
    mask_input.append(input_mask)
    mask_layer2 = Lambda(mask_func)(mask_input)
    x2 = layers.Conv1D(filters=25, kernel_size=3, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer2)
    # [?,seq_len,dim_conv]
    x2bn = layers.BatchNormalization()(x2)  #接收来自上一个卷积层的输出病进行归一化
    
    x2n = Lambda(plus)([x2bn,l_indrop])      #加入最初数据
    x2d = layers.Dropout(drop_hid)(x2n)   #随机失活
    mask_input = []
    mask_input.append(x2d)
    mask_input.append(input_mask)
    mask_layer3 = Lambda(mask_func)(mask_input)
    x3 = layers.Conv1D(filters=25, kernel_size=5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,bias_constraint=None)(mask_layer3)
    # [?,seq_len,dim_lstm]
    x3bn = layers.BatchNormalization()(x3)
    x3d = layers.Dropout(drop_hid)(x3bn)
    mask_input = []
    mask_input.append(x3d)
    mask_input.append(input_mask)
    mask_layer4 = Lambda(mask_func)(mask_input)
    x4 = layers.Conv1D(filters=25, kernel_size=9, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer4)
    # [?,seq_len,dim_lstm]
    x4bn = layers.BatchNormalization()(x4)
    x4n = Lambda(plus)([x4bn,l_indrop])
    x4d = layers.Dropout(drop_hid)(x4n)
    mask_input = []
    mask_input.append(x4n)
    mask_input.append(input_mask)
    mask_layer5 = Lambda(mask_func)(mask_input)
    x5 = layers.Conv1D(filters=25, kernel_size=15, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,bias_constraint=None)(mask_layer5)
    # [?,seq_len,dim_lstm]
    x5bn = layers.BatchNormalization()(x5)
    x5d = layers.Dropout(drop_hid)(x5bn)
    mask_input = []
    mask_input.append(x5d)
    mask_input.append(input_mask)
    mask_layer6 = Lambda(mask_func)(mask_input)
    x6 = layers.Conv1D(filters=25, kernel_size=21, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer6)
    # [?,seq_len,dim_lstm]
    x6bn = layers.BatchNormalization()(x6)
    x6n = Lambda(plus)([x6bn,l_indrop])
    x6d = layers.Dropout(drop_hid)(x6n)
    mask_input = []
    mask_input.append(x6n)
    mask_input.append(input_mask)
    mask_layer7 = Lambda(mask_func)(mask_input)
    
    x7 = layers.Conv1D(filters=25, kernel_size=3, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer7)
    
    x7bn = layers.BatchNormalization()(x7)
    x7d = layers.Dropout(drop_hid)(x7bn)
    mask_input = []
    mask_input.append(x6d)
    mask_input.append(input_mask)
    mask_layer8 = Lambda(mask_func)(mask_input)
    
    x8 = layers.Conv1D(filters=25, kernel_size=3, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer8)
    # [?,seq_len,dim_lstm]
    x8bn = layers.BatchNormalization()(x8)
    x8d = layers.Dropout(drop_hid)(x8bn)
    #x8l = layers.AveragePooling1D(pool_size=2, strides=1, padding='valid', data_format='channels_last')  #全局平均池化 输出维数不变
    mask_input = []
    mask_input.append(x8d)
    mask_input.append(input_mask)
    mask_layer9 = Lambda(mask_func)(mask_input)
    
  
    x9 = layers.Bidirectional(CuDNNGRU(dim_gru, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                                       return_sequences=True), merge_mode='sum')(mask_layer9)  
    # [?,seq_len,dim_lstm]
    x9bn = layers.BatchNormalization()(x9)
    x9d = layers.Dropout(drop_hid)(x9bn)
    mask_input = []
    mask_input.append(x9d)
    mask_input.append(input_mask)
    mask_layer10 = Lambda(mask_func)(mask_input)
    
    x10 = layers.Bidirectional(CuDNNGRU(dim_gru, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                                       return_sequences=True), merge_mode='sum')(mask_layer10)  
    # [?,seq_len,dim_lstm]
    x10bn = layers.BatchNormalization()(x10)
    x10d = layers.Dropout(drop_hid)(x10bn)
    mask_input = []
    mask_input.append(x10d)
    mask_input.append(input_mask)
    mask_layer11 = Lambda(mask_func)(mask_input)

     
    att = Attention(hidden=dim_gru, da=da, r=r, init='glorot_uniform', activation='tanh',
                    W1_regularizer=keras.regularizers.l2(W_regularizer),
                    W2_regularizer=keras.regularizers.l2(W_regularizer),
                    W1_constraint=None, W2_constraint=None, return_attention=True,
                    attention_regularizer_weight=Att_regularizer_weight)(
        layers.concatenate([mask_layer11, input_mask]))  # att=[?,r,dim_lstm]
    
    attbn = layers.BatchNormalization()(att[0])
    att_drop = layers.Dropout(drop_hid)(attbn)
    flat = layers.Flatten()(att_drop)
    flat_drop = layers.Dropout(drop_hid)(flat)
    lev1_output = layers.Dense(units=10 * 8, kernel_initializer='orthogonal', activation=None)(flat_drop)
    lev1_output_reshape = layers.Reshape([10, 8, 1])(lev1_output)
    lev1_output_bn = layers.BatchNormalization()(lev1_output_reshape)
    lev1_output_pre = layers.Activation('sigmoid')(lev1_output_bn)
    lev1_output_act = layers.Reshape([10,8],name='lev1')(lev1_output_pre)
    final = layers.MaxPooling2D(pool_size=[1, 8], strides=None, padding='same', data_format='channels_last')(
        lev1_output_pre)
    final = layers.Reshape([-1,],name='1ev2')(final)
    model_small = Model(inputs=[input, input_mask], outputs=[lev1_output_act, final])
    model_big = Model(inputs=[input, input_mask], outputs=[final])
    adam = optimizers.Adam(lr=lr)
    model_big.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model_small.compile(optimizer=adam, loss=['binary_crossentropy', 'binary_crossentropy'], metrics = ['accuracy'])
    model_big.summary()
    model_small.summary()
    return model_big, model_small

def process_input_train(seq_file,dir):
    processed_num=0
    if not os.path.exists(dir):
        os.mkdir(dir)
    for seq_record in list(SeqIO.parse(seq_file, "fasta")):
        processed_num+=1
        print("in loop, processing"+str(processed_num)+"\n")
        pssmfile=dir+seq_record.id+"_pssm.txt"
        inputfile=dir+seq_record.id+'_tem.fasta'
        seql = len(seq_record)
        if not os.path.exists(pssmfile):
            if os.path.exists(inputfile):
                os.remove(inputfile)
            SeqIO.write(seq_record, inputfile, 'fasta')
            psiblast_cline = NcbipsiblastCommandline(query=inputfile, db='D:/ncbi-blast-2.9.0+/db/swissprot/swissprot',                                     num_iterations=3, evalue=0.001, out_ascii_pssm=pssmfile, num_threads=4)
            stdout, stderr = psiblast_cline()
            os.remove(inputfile)

def process_input_user(seq_file,dir):
    processed_num=0
    if not os.path.exists(dir):
        os.mkdir(dir)
    for seq_record in list(SeqIO.parse(seq_file, "fasta")):
        processed_num+=1
        print("in loop, processing"+str(processed_num)+"\n")
        pssmfile=dir+seq_record.id+"_pssm.txt"
        inputfile=dir+seq_record.id+'_tem.fasta'
        seql = len(seq_record)
        if not os.path.exists(pssmfile):
            if os.path.exists(inputfile):
                os.remove(inputfile)
            SeqIO.write(seq_record, inputfile, 'fasta')
            try:
              psiblast_cline = NcbipsiblastCommandline(query=inputfile, 
                                         db='D:/ncbi-blast-2.9.0+/db/swissprot/swissprot',                                            num_iterations=3,evalue=0.001, out_ascii_pssm=pssmfile, num_threads=4)
              stdout, stderr = psiblast_cline()
              os.remove(inputfile)
            except:
              print("invalid protein: "+seq_record)


def var_model(train_x):
    [dim_gru, da, r, W_regularizer, Att_regularizer_weight, drop_per, drop_hid, lr] = [
        180, 369, 41, 0.00001,0.0007159, 0.1, 0.1, 0.0005]
    input = Input(shape=(train_x.shape[1:]), name="Input")  # input's shape=[?,seq_len,encoding_dim]
    input_mask = Input(shape=([train_x.shape[1], 1]), dtype='float32')  # (batch_size,max_len,1)
    l_indrop = layers.Dropout(drop_per)(input)

    mask_input = []
    mask_input.append(l_indrop)
    mask_input.append(input_mask)
    mask_layer1 = Lambda(mask_func)(mask_input)
    x1 = layers.Conv1D(filters=20, kernel_size=1, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer1)
   
    # [?,seq_len,dim_lstm]
    x1bn = layers.BatchNormalization()(x1)
    x1d = layers.Dropout(drop_hid)(x1bn)
    mask_input = []
    mask_input.append(x1d)
    mask_input.append(input_mask)
    mask_layer2 = Lambda(mask_func)(mask_input)
    x2 = layers.Conv1D(filters=20, kernel_size=3, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer2)
    # [?,seq_len,dim_lstm]
    x2bn = layers.BatchNormalization()(x2)
    x2d = layers.Dropout(drop_hid)(x2bn)
    mask_input = []
    mask_input.append(x2d)
    mask_input.append(input_mask)
    mask_layer3 = Lambda(mask_func)(mask_input)
    x3 = layers.Conv1D(filters=20, kernel_size=5, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer3)
    # [?,seq_len,dim_lstm]
    x3bn = layers.BatchNormalization()(x3)
    x3d = layers.Dropout(drop_hid)(x3bn)
    mask_input = []
    mask_input.append(x3d)
    mask_input.append(input_mask)
    mask_layer4 = Lambda(mask_func)(mask_input)
    x4 = layers.Conv1D(filters=20, kernel_size=9, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer4)
    # [?,seq_len,dim_lstm]
    x4bn = layers.BatchNormalization()(x4)
    x4d = layers.Dropout(drop_hid)(x4bn)
    mask_input = []
    mask_input.append(x4d)
    mask_input.append(input_mask)
    mask_layer5 = Lambda(mask_func)(mask_input)
    x5 = layers.Conv1D(filters=20, kernel_size=15, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer5)
    # [?,seq_len,dim_lstm]
    x5bn = layers.BatchNormalization()(x5)
    x5d = layers.Dropout(drop_hid)(x5bn)
    mask_input = []
    mask_input.append(x5d)
    mask_input.append(input_mask)
    mask_layer6 = Lambda(mask_func)(mask_input)
    x6 = layers.Conv1D(filters=20, kernel_size=21, strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=None,   
                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, 
                       bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mask_layer6)
    # [?,seq_len,dim_lstm]
    x6bn = layers.BatchNormalization()(x6)
    x6d = layers.Dropout(drop_hid)(x6bn)
    mask_input = []
    mask_input.append(x6d)
    mask_input.append(input_mask)
    mask_layer7 = Lambda(mask_func)(mask_input)
    
        
    x7 = layers.Bidirectional(CuDNNGRU(dim_gru, kernel_initializer="orthogonal", recurrent_initializer="orthogonal",
                                       return_sequences=True), merge_mode='sum')(mask_layer7)  
    # [?,seq_len,dim_lstm]
    x7bn = layers.BatchNormalization()(x7)
    x7d = layers.Dropout(drop_hid)(x7bn)
    mask_input = []
    mask_input.append(x7d)
    mask_input.append(input_mask)
    mask_layer8 = Lambda(mask_func)(mask_input)

    att = Attention(hidden=dim_gru, da=da, r=r, init='glorot_uniform', activation='tanh',
                    W1_regularizer=keras.regularizers.l2(W_regularizer),
                    W2_regularizer=keras.regularizers.l2(W_regularizer),
                    W1_constraint=None, W2_constraint=None, return_attention=True,
                    attention_regularizer_weight=Att_regularizer_weight)(
        layers.concatenate([mask_layer8, input_mask]))  # att=[?,r,dim_lstm]

    attbn = layers.BatchNormalization()(att[0])
    att_drop = layers.Dropout(drop_hid)(attbn)
    flat = layers.Flatten()(att_drop)
    flat_drop = layers.Dropout(drop_hid)(flat)
    lev1_output = layers.Dense(units=10, kernel_initializer='orthogonal', activation=None)(flat_drop)
    lev1_output_bn = layers.BatchNormalization()(lev1_output)
    lev1_output_act = layers.Activation('softmax')(lev1_output_bn)
    model = Model(inputs=[input, input_mask], outputs=lev1_output_act)
    adam = optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
