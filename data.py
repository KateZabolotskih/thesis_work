from matplotlib import cm

path_Kivu = "Kivu/"
path_North = "Permafrost/"
path_Baikal = "Baikal/"

files_Kivu = [
    "2.3_5", "2.4_7", "3.4_20", "4.2_80", "4.4_87", "4.5_108",
    "4.6_88", "5.3_66-1", "5.4_92", "5.6_95"
]
files_North = [
    "1701", "1702", "1704", "1706", "1711", "1712",
    "1727", "1728", "1729", "1730"
]
files_Baikal = [
    "1_25", "1_50"
]
file_type = ".txt"

color = cm.get_cmap('inferno')

amino_acids = {
    "C": {
        "Emission": [320, 30],
        "Excitation": [420, 60],
        "Color": 'r'
    },

    "A": {
        "Emission": [250, 10],
        "Excitation": [380, 100],
        "Color": 'g'
    },

    "M": {
        "Emission": [310, 10],
        "Excitation": [380, 40],
        "Color": 'br'
    },

    "B": {
        "Emission": [270, 10],
        "Excitation": [300, 20],
        "Color": 'y'
    },

    "T": {
        "Emission": [270, 10],
        "Excitation": [320, 30],
        "Color": 'g'
    }
}

areas = ["Kivu", "North", "Baikal"]

"""""
, "2_5", "2_25", "2_50", "2_75",
    "2_100", "2_125", "2_150", "2_215"
    
"""""