HUMAN_RAW_DATA_DIR = 'human_data/raw_data'
HUMAN_PROCESSED_DATA_DIR = 'human_data/processed_data'
HUMAN_ANALYZED_DATA_DIR = 'human_data/analyzed_data'
USER_STUDY_DIR = 'user_study_results'
PAT = {
    'p1': 'p1_eye_tracking_0818.tsv',
    'p2': 'p2_eye_tracking_0818.tsv',
    'p3': 'p3_eye_tracking_0827.tsv',
    'p4': 'p4_eye_tracking_0827.tsv',
    'p5': 'p5_eye_tracking_0901.tsv',
    'p6': 'p6_eye_tracking_0916.tsv',
    'p7': 'p7_eye_tracking_0916.tsv',
    'p8': 'p8_eye_tracking_0917.tsv',
    'p9': 'p9_eye_tracking_0917.tsv',
    'p10': 'p10_eye_tracking_0917.tsv',
    'p11': 'p11_eye_tracking_0918.tsv',
    'p12': 'p12_eye_tracking_0918.tsv',
    'p13': 'p13_eye_tracking_1018.tsv',
    'p14': 'p14_eye_tracking_1018.tsv',
    'p15': 'p15_eye_tracking_1018.tsv',
    'p16': 'p16_eye_tracking_1018.tsv',
    'p17': 'p17_eye_tracking_1018.tsv',
    'p18': 'p18_eye_tracking_1018.tsv',
    'p19': 'p19_eye_tracking_1018.tsv',
    'p20': 'p20_eye_tracking_1018.tsv',
    'p21': 'p21_eye_tracking_1018.tsv',
    'p22': 'p22_eye_tracking_1018.tsv',
    'p23': 'p23_eye_tracking_1018.tsv',
    'p24': 'p24_eye_tracking_1022.tsv',
    'p25': 'p25_eye_tracking_1022.tsv',
    'p26': 'p26_eye_tracking_1022.tsv',
    'p27': 'p27_eye_tracking_1022.tsv',
    'p28': 'p28_eye_tracking_1022.tsv',
    'p29': 'p29_eye_tracking_1022.tsv',
    'p30': 'p30_eye_tracking_1023.tsv',
    'p31': 'p31_eye_tracking_1023.tsv',
    'p32': 'p32_eye_tracking_1023.tsv',
}
FILENAME = 'Filename'

P_NAME = 'Participant name'
STIMU_NAME = 'Presented Stimulus name'
EYE_MOV_TYPE = 'Eye movement type'
GAZE_EVENT_DUR = 'Gaze event duration'
EYE_MOV_TYPE_IDX = 'Eye movement type index'
FIX_POINT_X = 'Fixation point X'
FIX_POINT_Y = 'Fixation point Y'
EYE_TRACKER_TIMESTAMP = 'Eyetracker timestamp'

SACCADE = "Saccade"
FIXATION = "Fixation"
UNCLASSIFIED = "Unclassified"
EYESNOTFOUND = "EyesNotFound"

SCREEN_WIDTH_CM = 52.8
SCREEN_HEIGHT_CM = 29.7
VIEWING_DISTANCE_CM = 50
SCREEN_RESOLUTION_WIDTH_PX = 1920
SCREEN_RESOLUTION_HEIGHT_PX = 1080

MERGE_PX_THRESHOLD = 17.65  # 17.65 for 0.5 degrees under our setting

STIM_1 = 'stim_1'
STIM_2 = 'stim_2'
STIM_3 = 'stim_3'
STIM_4 = 'stim_4'
STIM_5 = 'stim_5'
STIM_6 = 'stim_6'
STIM_7 = 'stim_7'
STIM_8 = 'stim_8'
STIM_9 = 'stim_9'
COND_THIRTY = 30
COND_SIXTY = 60
COND_NINETY = 90

SR = 1200  # Sampling rate, in Hz
SR_EPSILON = 100  # Sampling rate epsilon: tolerance for sampling rate difference, in Hz

METADATA = {
    '1': {
        FILENAME: PAT['p1'],
        P_NAME: 'p1',
        STIMU_NAME: {
            STIM_1: ['1', COND_THIRTY],
            STIM_2: ['2 (1)', COND_THIRTY],
            STIM_3: ['3 (1)', COND_THIRTY],
            STIM_4: ['4 (1)', COND_SIXTY],
            STIM_5: ['5 (1)', COND_SIXTY],
            STIM_6: ['6 (2)', COND_SIXTY],
            STIM_7: ['7 (2)', COND_NINETY],
            STIM_8: ['8 (1)', COND_NINETY],
            STIM_9: ['9 (1)', COND_NINETY],
        }
    },
    '2': {
        FILENAME: PAT['p2'],
        P_NAME: 'formal-p2-hao',
        STIMU_NAME: {
            STIM_1: ['1 (1)', COND_THIRTY],
            STIM_2: ['2 (2)', COND_THIRTY],
            STIM_3: ['3 (1)', COND_SIXTY],
            STIM_4: ['4 (1)', COND_SIXTY],
            STIM_5: ['5 (1)', COND_SIXTY],
            STIM_6: ['6 (2)', COND_NINETY],
            STIM_7: ['7 (2)', COND_NINETY],
            STIM_8: ['8 (1)', COND_NINETY],
            STIM_9: ['9 (1)', COND_THIRTY],
        }
    },
    '3': {
        FILENAME: PAT['p3'],
        P_NAME: 'Participant3-linfan',
        STIMU_NAME: {
            STIM_1: ['1', COND_SIXTY],
            STIM_2: ['2', COND_THIRTY],
            STIM_3: ['3', COND_THIRTY],
            STIM_4: ['4', COND_THIRTY],
            STIM_5: ['5', COND_NINETY],
            STIM_6: ['6', COND_NINETY],
            STIM_7: ['7', COND_NINETY],
            STIM_8: ['8', COND_SIXTY],
            STIM_9: ['9', COND_SIXTY],
        }
    },
    '4': {
        FILENAME: PAT['p4'],
        P_NAME: 'Participant4-yiqi',
        STIMU_NAME: {
            STIM_1: ['1', COND_NINETY],
            STIM_2: ['2', COND_NINETY],
            STIM_3: ['3', COND_NINETY],
            STIM_4: ['4', COND_THIRTY],
            STIM_5: ['5', COND_THIRTY],
            STIM_6: ['6', COND_THIRTY],
            STIM_7: ['7', COND_SIXTY],
            STIM_8: ['8', COND_SIXTY],
            STIM_9: ['9', COND_SIXTY],
        }
    },
    '5': {
        FILENAME: PAT['p5'],
        P_NAME: 'Participant5-janet',
        # P_NAME: 'Recording8',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_THIRTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '6': {
        FILENAME: PAT['p6'],
        P_NAME: 'Participant6-haoyu',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '7': {
        FILENAME: PAT['p7'],
        P_NAME: 'Participant7',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_NINETY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_SIXTY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '8': {
        FILENAME: PAT['p8'],
        P_NAME: 'Participant8',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '9': {
        FILENAME: PAT['p9'],
        P_NAME: 'Participant9',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '10': {
        FILENAME: PAT['p10'],
        P_NAME: 'Participant10',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '11': {
        FILENAME: PAT['p11'],
        P_NAME: 'Participant11',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_SIXTY],
            STIM_7: ['stim_7', COND_SIXTY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '12': {
        FILENAME: PAT['p12'],
        P_NAME: 'Participant12',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '13': {
        FILENAME: PAT['p13'],
        P_NAME: 'p13',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '14': {
        FILENAME: PAT['p14'],
        P_NAME: 'p14',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '15': {
        FILENAME: PAT['p15'],
        P_NAME: 'p15',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_THIRTY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_NINETY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_SIXTY],
            STIM_7: ['stim_7', COND_SIXTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '16': {
        FILENAME: PAT['p16'],
        P_NAME: 'p16',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_THIRTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_SIXTY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '17': {
        FILENAME: PAT['p17'],
        P_NAME: 'p17',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_NINETY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '18': {
        FILENAME: PAT['p18'],
        P_NAME: 'p18',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_THIRTY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_NINETY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_SIXTY],
            STIM_7: ['stim_7', COND_SIXTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '19': {
        FILENAME: PAT['p19'],
        P_NAME: 'p19',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_THIRTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_SIXTY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '20': {
        FILENAME: PAT['p20'],
        P_NAME: 'p20',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_THIRTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_NINETY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_SIXTY],
            STIM_7: ['stim_7', COND_SIXTY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '21': {
        FILENAME: PAT['p21'],
        P_NAME: 'p21',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '22': {
        FILENAME: PAT['p22'],
        P_NAME: 'p22',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_NINETY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_SIXTY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '23': {
        FILENAME: PAT['p23'],
        P_NAME: 'p23',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_THIRTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '24': {
        FILENAME: PAT['p24'],
        P_NAME: 'p24',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '25': {
        FILENAME: PAT['p25'],
        P_NAME: 'p25',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_THIRTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '26': {
        FILENAME: PAT['p26'],
        P_NAME: 'p26',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_NINETY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_SIXTY],
        }
    },
    '27': {
        FILENAME: PAT['p27'],
        P_NAME: 'p27',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '28': {
        FILENAME: PAT['p28'],
        P_NAME: 'p28',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '29': {
        FILENAME: PAT['p29'],
        P_NAME: 'p29',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_THIRTY],
            STIM_4: ['stim_4', COND_THIRTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_SIXTY],
            STIM_7: ['stim_7', COND_SIXTY],
            STIM_8: ['stim_8', COND_SIXTY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '30': {
        FILENAME: PAT['p30'],
        P_NAME: 'p30',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_THIRTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_NINETY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
    '31': {
        FILENAME: PAT['p31'],
        P_NAME: 'p31',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_SIXTY],
            STIM_2: ['stim_2', COND_SIXTY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_NINETY],
            STIM_5: ['stim_5', COND_NINETY],
            STIM_6: ['stim_6', COND_NINETY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_THIRTY],
        }
    },
    '32': {
        FILENAME: PAT['p32'],
        P_NAME: 'p32',
        STIMU_NAME: {
            STIM_1: ['stim_1', COND_NINETY],
            STIM_2: ['stim_2', COND_NINETY],
            STIM_3: ['stim_3', COND_SIXTY],
            STIM_4: ['stim_4', COND_SIXTY],
            STIM_5: ['stim_5', COND_SIXTY],
            STIM_6: ['stim_6', COND_THIRTY],
            STIM_7: ['stim_7', COND_THIRTY],
            STIM_8: ['stim_8', COND_THIRTY],
            STIM_9: ['stim_9', COND_NINETY],
        }
    },
}

# Output metrics
PID = 'Participant ID'
EPSID = 'Episode ID'
STIM_ID = 'Stimulus ID'
TRIAL_COND = 'Trial Condition'
FIX_COUNT = 'Fixation Count'
AVG_FIX_COUNT_PER_SEC = 'Average Fixation Count per Second'
AVG_SACCADE_COUNT_PER_SEC = 'Average Saccade Count per Second'
TOTAL_FIX_DUR = 'Total Fixation Duration'
AVG_FIX_DUR_PER_SEC = 'Average Fixation Duration per Second'
AVG_FIX_DUR_PER_COUNT = 'Average Fixation Duration per Count'
SACCADE_COUNT = 'Saccade Count'
FIX_COUNT_PERCENTAGE = 'Fixation Count Percentage'
AVG_SACCADE_LENGTH_PX = 'Average Saccade Length (px)'
AVG_SACCADE_LENGTH_DEG = 'Average Saccade Length (deg)'
AVG_SACCADE_VEL_PX = 'Average Saccade Velocity (px/s)'
AVG_SACCADE_VEL_DEG = 'Average Saccade Velocity (deg/s)'
REGRESSION_FREQ_XY = 'Regression Frequency XY'
REGRESSION_FREQ_X = 'Regression Frequency X'
WORD_SKIP_PERCENTAGE_BY_READING_PROGRESS = 'Word Skip Percentage by Reading Progress'
WORD_SKIP_PERCENTAGE_BY_SACCADES = 'Word Skip Percentage by Saccades'
WORD_SKIP_PERCENTAGE_BY_SACCADES_V2 = 'Word Skip Percentage by Saccades V2 With Word Index Correction'
WORD_NOT_COVERED_PERCENTAGE = 'Word Not Covered Percentage'
REVISIT_PERCENTAGE_BY_READING_PROGRESS = 'Revisit Percentage by Reading Progress'
REVISIT_PERCENTAGE_BY_SACCADES = 'Revisit Percentage by Saccades'
REVISIT_PERCENTAGE_BY_SACCADES_V2 = 'Revisit Percentage by Saccades V2 With Word Index Correction'
MCQ_ACC = 'MCQ Accuracy'
FREE_RECALL_SCORE = 'Free Recall Score'
READING_SPEED = 'Average Reading Speed (wpm)'
NUM_FIXATIONS = 'Number of Fixations'

# Scanpath-related metrics
LEV = 'Levenshtein Distance'
NLD = 'Normalized Levenshtein Distance'
DTW = 'DTW Distance'
LCS = "LCS Length"
FRECT = "Frechet Distance"
HAUSDF = "Hausdorff Distance"
ScanMatchScore = "ScanMatch Score"
TDE = "TDE Distance"
RR = "Recurrence Rate"
MD = "Mann Distance"
EYEALS = "Eyeanalysis Distance"
DET = "Determinism"
LAM = "Laminarity"

EPSILON = 1e-6

# Bounding box metadata dir
# ----------------------------------------------------------------------------------------------------------------------
BBOX_METADATA_DIR = '/home/baiy4/reading-model/step5/data/gen_envs/10_27_15_58_100_images_W1920H1080WS16_LS40_MARGIN400/simulate/metadata.json'

# Color hues
# ----------------------------------------------------------------------------------------------------------------------
HUMAN_DATA_COLOR = (0, 0.4470, 0.7410)
SIMULATION_RESULTS_COLOR = (0.4660, 0.6740, 0.1880)

# Metrics Calculation
# ----------------------------------------------------------------------------------------------------------------------
MODE_COORD = 'coord'
MODE_WORD_INDEX = 'word_index'

# Baseline Models
# ----------------------------------------------------------------------------------------------------------------------
SCANPATH_VQA = 'Scanpath_VQA'
EZREADER = 'EZReader'
SWIFT = 'SWIFT'
SCANDL = 'ScanDL'
EYETTENTION = 'Eyettention'

# Parameters for the comprehension tests
# ----------------------------------------------------------------------------------------------------------------------
TEXT_SIMILARITY_THRESHOLD = "text_similarity_threshold"
EXPLORATION_RATE = "exploration_rate"

# Metrics for the comprehension tests
# ----------------------------------------------------------------------------------------------------------------------
MCQ_ACC = 'MCQ Accuracy'
FREE_RECALL_SCORE = 'Free Recall Score'

# Conditions for the comprehension tests
# ----------------------------------------------------------------------------------------------------------------------
TIME_CONSTRAINTS = 'time_constraint'