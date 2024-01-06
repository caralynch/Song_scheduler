import numpy as np
import math
import pandas as pd 
import sys
import warnings
warnings.simplefilter('ignore')

# Read in input matrix
if len(sys.argv) > 1:
    infile = sys.argv[1]
else:
    infile = 'matrix3.csv'

if len(sys.argv) > 2:
    outfile = sys.argv[2]
else:
    outfile = 'successive_pairs.xlsx'

df = pd.read_csv(infile)

# Formatting input

def format_values(df):
    # remove rows with 0 rehearsal duration
    df = df.loc[df.slot_size > 0, :]

    # change cast strings to sets
    df.cast = (df.cast.str.split(", ")).apply(set)

    # round durations to nearest 0.25 of a rehearsal (30 mins)
    df.loc[df.slot_size % 0.25 != 0, "slot_size"] = df.loc[df.slot_size % 0.25 != 0].slot_size.apply(lambda x: math.ceil(x*4)/4)

    # remove trailing whitespaces from rehearsal lead entries
    df.loc[:, "lead"] = df.loc[:, "lead"].str.strip()

    return df

# Rename columns
to_rename = {
    'Song': 'song_name',
    'Duration': 'slot_size',
    'Lead': 'lead',
    'Singers': 'cast',
}
df.rename(columns=to_rename, inplace=True)

# format all col values as required
df = format_values(df)

# Find the types of rehearsals listed (so these can occur simultaneously)
rehearsal_leads = df.lead.unique()

def get_full_and_partial_slots(df, rehearsal_lead):
    """Separate out rehearsals by rehearsal lead and full and partial slots (if a slot
    is partial then it needs to be paired with another slot to make a full slot)
    """
    to_rehearse = df[df.lead == rehearsal_lead]
    separated_slots = {
        'full_slots': to_rehearse[to_rehearse.slot_size % 1 == 0],
        'partial_slots': to_rehearse[to_rehearse.slot_size % 1 != 0]
    }
    return separated_slots

def set_overlap(sets):
    for i, set in enumerate(sets):
        if  i == 0:
            set_intersections = set
            set_unions = set
        else:
            set_intersections = set_intersections & set
            set_unions = set_intersections | set
    return len(set_intersections)/len(set_unions)


def calculate_overlap_array(partial_slots, dimensions):
    
    overlap_array = np.zeros([len(partial_slots)]*dimensions)

    for i in range(len(partial_slots)):
        for j in range(len(partial_slots)):
            if i == j:
                overlap_array[i,j] = -1
            else:
                if dimensions == 2:
                    overlap_array[i,j] = set_overlap([partial_slots.loc[i, 'cast'], partial_slots.loc[j, "cast"]])
                else:
                    for k in range(len(partial_slots)):
                        if (i == k) or (j == k):
                            overlap_array[i,j,k] == -1
                        else:
                            if dimensions == 3:
                                overlap_array[i,j,k] = set_overlap([partial_slots.loc[i, 'cast'], partial_slots.loc[j, "cast"], partial_slots.loc[k, "cast"]])
                            else:
                                for l in range(overlap_array.shape[3]):
                                    if (i==l) or (j==l) or (k==l):
                                        overlap_array[i,j,k,l] == -1
                                    else:
                                        overlap_array[i,j,k,l] = set_overlap([partial_slots.loc[i, 'cast'], partial_slots.loc[j, "cast"], partial_slots.loc[k, "cast"], partial_slots.loc[l, "cast"]])
    return(overlap_array)

def find_complementary_durations(song_slot_size, song_name, remaining_slots):
    complementary_durations = [
        x for x in remaining_slots.slot_size.unique() if (x + song_slot_size) % 1 == 0
        ]
    # if can't find pair, tries trios
    if not complementary_durations:
        #print(f"No time match for {song_name}, recomputing with a trio")
        for i in remaining_slots.slot_size.unique():
            
            for j in remaining_slots.slot_size.unique():
                
                if (i + j + song_slot_size) % 1 == 0:
                    if ((i,j) not in complementary_durations) and ((j,i) not in complementary_durations):
                        complementary_durations.append((i,j))
        if not complementary_durations:
            #print(f"No time match for {song_name}, recomputing with a quartet")
            for i in remaining_slots.slot_size.unique():

                for j in remaining_slots.slot_size.unique():

                        for k in remaining_slots.slot_size.unique():

                            if (i + j + k + song_slot_size) % 1 == 0:
                                if (((i,j,k) not in complementary_durations) and ((i,k,j) not in complementary_durations) and
                                    ((j,k,i) not in complementary_durations) and ((j,i,k) not in complementary_durations) and
                                    ((k,i,j) not in complementary_durations) and ((k,j,i) not in complementary_durations)):
                                        complementary_durations.append((i,j,k))
            if not complementary_durations:
                print(f"No time match for {song_name}")
    return complementary_durations

def find_best_song_tuple_from_complementary_durations(song_index,
                                                complementary_durations,
                                                partial_slots,
                                                remaining_slots,
                                                overlap_array
                                                ):
    # check if complementary durations are floats (single pairs) or sets (trios, quartets)
    if isinstance(complementary_durations[0], float):
        set_size = 2
    else:
        set_size = len(complementary_durations[0]) + 1


    song_sets = []
    overlap_scores = []

    for complementary_duration in complementary_durations:
        # if it's a float, then there's only a pair, yay
        if set_size == 2:
            possible_matches = remaining_slots[remaining_slots.slot_size == complementary_duration].index
            if list(possible_matches):
                max_overlap = overlap_array.loc[possible_matches, song_index].max()
                best_choice = overlap_array.loc[possible_matches, song_index].idxmax()
                song_sets.append((song_index, best_choice))
                overlap_scores.append(max_overlap)
                #remaining_slots.drop(index=best_choice, inplace=True)
        else:
            song_subset = [song_index]
            for duration in complementary_duration:
                possible_matches = remaining_slots[remaining_slots.slot_size == duration].index
                best_choice = overlap_array.loc[possible_matches, song_index].idxmax()
                song_subset.append(best_choice)
                #remaining_slots.drop(index=best_choice, inplace=True)
            song_sets.append(tuple(song_subset))
            cast_sets = [partial_slots.loc[x, 'cast'] for x in song_subset]
            overlap_scores.append(set_overlap(cast_sets))
    
    if overlap_scores:
        set_with_max_overlap = song_sets[np.argmin(overlap_scores)]
    else:
        set_with_max_overlap = None

    return set_with_max_overlap

def get_set_vals_from_tuple(set_tuple, partial_slots):
    current_set = set()
    duration = 0
    cast = set()
    for element in set_tuple:
        current_set.add(partial_slots.loc[element, "original_index"])
        duration += partial_slots.loc[element, "slot_size"]
        cast = cast | partial_slots.loc[element, "cast"]
    
    out_dict = {
        'pair_indices': current_set,
        'slot_size': duration,
        'cast': cast
    }

    return out_dict

def find_partial_slot_pairings(partial_slots):

    # order songs by cast size (so large numbers are addressed first)
    partial_slots['cast_size'] = partial_slots.cast.apply(len)
    partial_slots = partial_slots.sort_values(by='cast_size', ascending=False).reset_index().rename(columns={'index': 'original_index'})

    # get pair overlap arrays - start with only 2d and make 3 and 4d only if required
    overlap_array = pd.DataFrame(calculate_overlap_array(partial_slots, 2))

    # Create paired and unpaired slot dictionaries

    paired_slots = {
        'pair_indices': [],
        'slot_size': [],
        'cast': []
    }

    unpaired_slots = {
        'pair_indices': [],
        'slot_size': [],
        'cast': []
    }

    # Create a remain_to_pair list so that only unpaired songs are considered when pairing
    remain_to_pair = list(partial_slots.sort_values(by="slot_size", ascending=False).index)

    # Iterate through songs to pair
    while len(remain_to_pair) > 1:
        song_index = remain_to_pair[0]
        remain_to_pair.remove(song_index)
        remaining_slots = partial_slots.loc[remain_to_pair, :]
        original_song_index = partial_slots.loc[song_index, 'original_index']
        song_slot_size = partial_slots.loc[song_index, 'slot_size']
        song_name = partial_slots.loc[song_index, 'song_name']

        complementary_durations  = find_complementary_durations(song_slot_size, song_name, remaining_slots)

        # if not complementary durations were found, add this song to the unpaired dict
        if not complementary_durations:
            print(f"No time match for {song_name}")
            unpaired_slots['pair_indices'].append({original_song_index})
            unpaired_slots['slot_size'].append(song_slot_size)
            unpaired_slots['cast'].append(partial_slots.loc[song_index, 'cast'])
        # if complementary durations were found, try pairing the song
        else:
            best_set_tuple = find_best_song_tuple_from_complementary_durations(song_index, complementary_durations, partial_slots, remaining_slots, overlap_array)
            if best_set_tuple is None:
                unpaired_slots['pair_indices'].append({original_song_index})
                unpaired_slots['slot_size'].append(song_slot_size)
                unpaired_slots['cast'].append(partial_slots.loc[song_index, 'cast'])
            else:
                
                set_tuple_dict = get_set_vals_from_tuple(best_set_tuple, partial_slots)
                if set_tuple_dict['pair_indices'] in paired_slots['pair_indices']:
                    pass
                else:
                    for key in paired_slots:
                        paired_slots[key].append(set_tuple_dict[key])
                    for elem in best_set_tuple:
                        if elem in remain_to_pair:
                            remain_to_pair.remove(elem)
    if len(remain_to_pair) == 1:
        song_index = remain_to_pair[0]
        remain_to_pair.remove(song_index)
        remaining_slots = partial_slots.loc[remain_to_pair, :]
        original_song_index = partial_slots.loc[song_index, 'original_index']
        song_slot_size = partial_slots.loc[song_index, 'slot_size']
        song_name = partial_slots.loc[song_index, 'song_name']
        print(f"No time match for {song_name}")
        unpaired_slots['pair_indices'].append({original_song_index})
        unpaired_slots['slot_size'].append(song_slot_size)
        unpaired_slots['cast'].append(partial_slots.loc[song_index, 'cast'])
    
    out_dict = {
        'paired': pd.DataFrame.from_dict(paired_slots),
        'unpaired': pd.DataFrame.from_dict(unpaired_slots)
    }

    return out_dict

def single_set(a):
    return {a}


rehearsals = {}
for rehearsal_lead in rehearsal_leads:
    separated_slots = get_full_and_partial_slots(df, rehearsal_lead)

    # full slots are fine, need to find pairings for partial slots
    rehearsals[rehearsal_lead] = find_partial_slot_pairings(separated_slots['partial_slots'])
    full_slots = separated_slots['full_slots'].reset_index().rename(columns={'index': 'pair_indices'})
    full_slots.pair_indices = full_slots.pair_indices.apply(single_set)
    rehearsals[rehearsal_lead]['full_slots'] = full_slots

full_song_lists = {}
for lead in rehearsals:

    for i, key in enumerate(rehearsals[lead].keys()):
        if i == 0:
            to_concat = rehearsals[lead][key]
        else:
            to_concat = pd.concat((to_concat, rehearsals[lead][key]), join='inner')
    full_song_lists[lead] = to_concat.reset_index(drop=True)

index_lookup = df.song_name

def map_set_to_song_names(pair_set):
    song_names = ''
    for i in pair_set:
        song_names += f"{index_lookup[i]}, "
    song_names = song_names.strip(', ')
    return song_names

for lead in full_song_lists:
    full_song_lists[lead]['songs'] = full_song_lists[lead].pair_indices.apply(map_set_to_song_names)

simultaneous_options = full_song_lists.copy()


for lead in simultaneous_options:
    new_cols = {}
    for lead2 in [x for x in simultaneous_options if x!=lead]:
        new_cols[lead2] = []
    for i in simultaneous_options[lead].index:
        for lead2 in [x for x in simultaneous_options if x!=lead]:
            possible_pairs = []
            for j in simultaneous_options[lead2].index:
                if len(simultaneous_options[lead].loc[i, 'cast'] & simultaneous_options[lead2].loc[j,'cast']) == 0:
                    possible_pairs.append(simultaneous_options[lead2].loc[j, 'pair_indices'])
            new_cols[lead2].append(possible_pairs)
    for lead2 in new_cols:
        simultaneous_options[lead][f"{lead2}_possible_pairs"] = new_cols[lead2]

with pd.ExcelWriter(outfile) as excel_writer:
    df.to_excel(excel_writer, sheet_name='Song_lookup')
    for lead in simultaneous_options:
        simultaneous_options[lead].to_excel(excel_writer, sheet_name=lead, index=False)