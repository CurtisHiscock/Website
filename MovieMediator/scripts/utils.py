# utils.py
##########
import numpy as np

#Function that gets the director's name, NaN if not listed
#--------------------------------------------------------
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


#Return top 3 elements or fewer
#----------------------------------------------------------------
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names[:3] if len(names) > 3 else names
    return []


#Clean strings (lowercase, no spaces)
#------------------------------------------------------------------------
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ''


#Build metadata soup
#-------------
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])