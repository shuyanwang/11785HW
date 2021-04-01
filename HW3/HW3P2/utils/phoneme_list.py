N_PHONEMES = 41
PHONEME_LIST = [
    " ",
    "SIL",
    "SPN",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "H",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH"
]

PHONEME_MAP = [
    " ",
    ".",  # SIL
    "!",  # SPN
    "a",  # AA
    "A",  # AE
    "h",  # AH
    "o",  # AO
    "w",  # AW
    "y",  # AY
    "b",  # B
    "c",  # CH
    "d",  # D
    "D",  # DH
    "e",  # EH
    "r",  # ER
    "E",  # EY
    "f",  # F
    "g",  # G
    "H",  # H
    "i",  # IH
    "I",  # IY
    "j",  # JH
    "k",  # K
    "l",  # L
    "m",  # M
    "n",  # N
    "N",  # NG
    "O",  # OW
    "Y",  # OY
    "p",  # P
    "R",  # R
    "s",  # S
    "S",  # SH
    "t",  # T
    "T",  # TH
    "u",  # UH
    "U",  # UW
    "v",  # V
    "W",  # W
    "?",  # Y
    "z",  # Z
    "Z"  # ZH
]

assert len(PHONEME_LIST) == len(PHONEME_MAP)
assert len(set(PHONEME_MAP)) == len(PHONEME_MAP)

if __name__ == '__main__':
    print(len(PHONEME_MAP))
