

import pandas as pd
import os
import re

regex_punctuation  = re.compile('[\',\.\-/\n]')
regex_alphanum     = re.compile('[^a-zA-Z0-9_ ]')
regex_num          = re.compile('\d[\d ]+')
regex_sectionbreak = re.compile('____+')


def bin_age(age):
    age = int(age)
    if age < 20:
        return 'AGE_LESS_THAN_TWENTY'
    if age < 30:
        return 'AGE_BETWEEN_TWENTY_AND_THIRTY'
    if age < 40:
        return 'AGE_BETWEEN_THIRTY_AND_FOURTY'
    if age < 50:
        return 'AGE_BETWEEN_FOURTY_AND_FIFTY'
    if age < 60:
        return 'AGE_BETWEEN_FIFTY_AND_SIXTY'
    if age < 70:
        return 'AGE_BETWEEN_SIXTY_AND_SEVENTY'
    if age < 80:
        return 'AGE_BETWEEN_SEVENTY_AND_EIGHTY'
    if age < 90:
        return 'AGE_BETWEEN_EIGHTY_AND_NINETY'
    if age > 90:
        return 'AGE_OVER_NINETY'
    raise Exception('shouldnt get here')

def tokenize(text):
    text = text.strip()

    # remove phi tags
    tags = re.findall('\[\*\*.*?\*\*\]', text)
    for tag in set(tags):
        text = text.replace(tag, ' ')

    # collapse phrases (including diagnoses) into single tokens
    if text != text.upper():
        caps_matches = set(re.findall('([A-Z][A-Z_ ]+[A-Z])', text))
        for caps_match in caps_matches:
            caps_match = re.sub(' +', ' ', caps_match)
            if len(caps_match) < 35:
                replacement = caps_match.replace(' ','_')
                text = text.replace(caps_match,replacement)

    year_regexes = ['(\d+) years? old', '\s(\d+) ?yo ', '(\d+)[ -]year-old',
                    '\s(\d+) yr old', '\s(\d+) yo[m/f]', '(\d+) y/o ']
    year_text = ' ' + text.lower()
    for year_regex in year_regexes:
        year_matches = re.findall(year_regex, year_text)
        for match in set(year_matches):
            binned_age = ' %s ' % bin_age(match)
            text = text.replace(match, binned_age)

    text = re.sub('_+', '_', text)

    text = text.lower()
    text = re.sub(regex_punctuation , ' '  , text)
    text = re.sub(regex_alphanum    , ''   , text)
    text = re.sub(regex_num         , ' 0 ', text)
    return text.strip().split()


def get_kexin_data(folder, stage , datatype, test=False):
    homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    homedir = os.path.dirname(homedir)
    # homedir = os.path.dirname(homedir)


    filename = '%s/clinicalBERT/data/good_datasets/%s/%s/%s.csv' % (homedir,folder, stage, datatype)
    admission_filename = '%s/clinicalBERT/data/ADMISSIONS.csv' % (homedir)
    if test:
        df = pd.read_csv(filename, nrows=100 )
    else:
        df = pd.read_csv(filename)
    admissions = pd.read_csv(admission_filename)
    merged = pd.merge(df, admissions, left_on=['ID'], right_on=['HADM_ID'], how='left')

    X = dict()
    Y = dict()
    # level = 'HDM'
    for idx, val in merged.iterrows():

        if val['SUBJECT_ID'] not in X:
            X[val['SUBJECT_ID']]  = [(None, stage, tokenize(val['TEXT']))]
            Y[val['SUBJECT_ID']]  = {'readmission': int(val['Label'])}
        else:
            X[val['SUBJECT_ID']].append((None, stage, tokenize(val['TEXT'])))

    assert sorted(X.keys()) == sorted(Y.keys())
    return X, Y



def get_kexin_data_v2(folder, stage , datatype, test=False):
    homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    homedir = os.path.dirname(homedir)
    # homedir = os.path.dirname(homedir)


    filename = '%s/clinicalBERT/data/good_datasets/%s/%s/%s.csv' % (homedir,folder, stage, datatype)
    admission_filename = '%s/clinicalBERT/data/ADMISSIONS.csv' % (homedir)
    if test:
        df = pd.read_csv(filename, nrows=100 )
    else:
        df = pd.read_csv(filename)
    # admissions = pd.read_csv(admission_filename)
    # merged = pd.merge(df, admissions, left_on=['ID'], right_on=['HADM_ID'], how='left')

    X = dict()
    Y = dict()

    for idx, val in df.iterrows():

        if val['ID'] not in X:
            X[val['ID']]  = [(None, stage, tokenize(val['TEXT']))]
            Y[val['ID']]  = {'readmission': int(val['Label'])}
        else:
            X[val['ID']].append((None, stage, tokenize(val['TEXT'])))

    assert sorted(X.keys()) == sorted(Y.keys())
    return X, Y