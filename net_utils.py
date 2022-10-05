import tensorflow as tf
import math
tf.math.pi = math.pi
def nll(y_true,y_hat):
    raw_mu,var_values = tf.split(y_hat,2,-1)
    y_diff     = tf.subtract(raw_mu,y_true)
    loss       =  tf.reduce_mean(0.5*tf.math.log(var_values  + 1e-6 ) + 0.5*tf.divide(tf.square(y_diff), var_values  + 1e-6)) + 0.5*tf.math.log(2*tf.math.pi)
    return loss    

def mse(y_true,y_hat):
    raw_mu,var_values = tf.split(y_hat,2,-1)
    return tf.keras.losses.MeanSquaredError()(y_true,raw_mu)

def log_var(y_true,y_hat):
    raw_mu,var_values = tf.split(y_hat,2,-1)
    return tf.reduce_mean(tf.math.log(var_values  + 1e-6 ))

_map = {0: 'oocytes_trisopterus_states_5b',
 1: 'pittsburg-bridges-SPAN',
 2: 'statlog-heart',
 3: 'molec-biol-promoter',
 4: 'yeast',
 5: 'monks-3',
 6: 'titanic',
 7: 'synthetic-control',
 8: 'ionosphere',
 9: 'pittsburg-bridges-T-OR-D',
 10: 'breast-tissue',
 11: 'ecoli',
 12: 'oocytes_merluccius_nucleus_4d',
 13: 'plant-margin',
 14: 'conn-bench-vowel-deterding',
 15: 'optical',
 16: 'magic',
 17: 'miniboone',
 18: 'heart-switzerland',
 19: 'breast-cancer-wisc-prog',
 20: 'ringnorm',
 21: 'lung-cancer',
 22: 'steel-plates',
 23: 'plant-shape',
 24: 'echocardiogram',
 25: 'lymphography',
 26: 'energy-y2',
 27: 'musk-1',
 28: 'plant-texture',
 29: 'statlog-australian-credit',
 30: 'vertebral-column-2clases',
 31: 'abalone',
 32: 'blood',
 33: 'credit-approval',
 34: 'molec-biol-splice',
 35: 'wine-quality-white',
 36: 'bank',
 37: 'car',
 38: 'low-res-spect',
 39: 'horse-colic',
 40: 'hill-valley',
 41: 'statlog-shuttle',
 42: 'hayes-roth',
 43: 'cardiotocography-3clases',
 44: 'breast-cancer-wisc',
 45: 'adult',
 46: 'glass',
 47: 'fertility',
 48: 'mammographic',
 49: 'statlog-german-credit',
 50: 'oocytes_merluccius_states_2f',
 51: 'congressional-voting',
 52: 'soybean',
 53: 'planning',
 54: 'pittsburg-bridges-MATERIAL',
 55: 'statlog-vehicle',
 56: 'zoo',
 57: 'arrhythmia',
 58: 'lenses',
 59: 'ozone',
 60: 'seeds',
 61: 'cylinder-bands',
 62: 'wine',
 63: 'tic-tac-toe',
 64: 'acute-nephritis',
 65: 'connect-4',
 66: 'pima',
 67: 'statlog-image',
 68: 'chess-krvkp',
 69: 'musk-2',
 70: 'waveform',
 71: 'flags',
 72: 'wall-following',
 73: 'pendigits',
 74: 'iris',
 75: 'cardiotocography-10clases',
 76: 'statlog-landsat',
 77: 'twonorm',
 78: 'heart-cleveland',
 79: 'primary-tumor',
 80: 'oocytes_trisopterus_nucleus_2f',
 81: 'post-operative',
 82: 'spect',
 83: 'acute-inflammation',
 84: 'chess-krvk',
 85: 'dermatology',
 86: 'libras',
 87: 'mushroom',
 88: 'parkinsons',
 89: 'waveform-noise',
 90: 'heart-hungarian',
 91: 'heart-va',
 92: 'audiology-std',
 93: 'haberman-survival',
 94: 'energy-y1',
 95: 'page-blocks',
 96: 'conn-bench-sonar-mines-rocks',
 97: 'semeion',
 98: 'hepatitis',
 99: 'contrac',
 100: 'led-display',
 101: 'breast-cancer-wisc-diag',
 102: 'vertebral-column-3clases',
 103: 'ilpd-indian-liver',
 104: 'monks-1',
 105: 'image-segmentation',
 106: 'pittsburg-bridges-TYPE',
 107: 'thyroid',
 108: 'nursery',
 109: 'wine-quality-red',
 110: 'breast-cancer',
 111: 'letter',
 112: 'pittsburg-bridges-REL-L',
 113: 'monks-2',
 114: 'balloons',
 115: 'spectf',
 116: 'balance-scale',
 117: 'teaching',
 118: 'spambase',
 119: 'annealing'}