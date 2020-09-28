# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

!git clone https://github.com/iitmnlp/indic-swipe.git
data=pd.read_excel('/content/indic-swipe/indic-to-indic-datasets/Gujrati.xlsx')

lang = 'Gujarati'

row_1 = ['અ', 'આ', 'ઇ', 'ઈ', 'ઉ', 'ઊ', 'ઋ', 'એ', 'ઐ', 'ઓ', 'ઔ']
row_2 = ['ક', 'ખ', 'ગ', 'ઘ', 'ઙ', 'ચ', 'છ', 'જ', 'ઝ', 'ઞ', '઼']
row_3 = ['ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 'ત', 'થ', 'દ', 'ધ', 'ન', 'ં']
row_4 = ['પ', 'ફ', 'બ', 'ભ', 'મ', 'ય', 'ર', 'લ', 'ળ', 'વ', 'શ']
row_5 = ['ષ', 'સ', 'હ',',','N','D','D','D','ઁ', '્','D'] 

matras = [ 'ા',	'િ', 'ી', 'ુ', 'ૂ', 'ૃ', 'ે', 'ૈ', 'ો', 'ૌ']

char_set = row_1+row_2+row_3+row_4+row_5+matras

indices_to_remove = []
for i in range(len(data['indic'])):
    if any(elem not in char_set for elem in data['indic'][i]) or all(j==data['indic'][i][0] for j in data['indic'][i]):
        indices_to_remove.append(i)


#print("Number of words before filtering = ",len(data))

data.drop(indices_to_remove, inplace=True)

#print("Number of words after filtering = ",len(data))

keyboard_char_len = 11
keyboard_char_wid = 5

samples_per_word = 1
num_characters_on_keyboard = 55

keyboard_rows = [row_1,row_2,row_3,row_4, row_5]

valid_chars_dict={}
list_of_all_valid_chars = row_1+row_2+row_3+row_4+row_5+['<e>']
list_of_all_valid_chars = list(set(list_of_all_valid_chars))
list_of_all_valid_chars.sort()
for i in range(len(list_of_all_valid_chars)):
    if not list_of_all_valid_chars[i] in valid_chars_dict:
        valid_chars_dict[list_of_all_valid_chars[i]]=i

for i in range(len(matras)):
    valid_chars_dict[matras[i]] = valid_chars_dict[(row_1+['ऑ'])[i+1]]

num_characters_on_keyboard_for_ctc_model = len(list_of_all_valid_chars)-1
#print("num_characters_on_keyboard_for_ctc_model = ",num_characters_on_keyboard_for_ctc_model)

actual_keyboard = np.array(keyboard_rows)
#print(actual_keyboard)

def make_keyboard(keyboard_rows, res):
    board = np.zeros((15,30), dtype='<U1')
    for k in range(5):
        row = keyboard_rows[k]
        for i in range(len(row)):
            board[k*res:(k+1)*res,res*i:res*(i+1)] = row[i]
    return board

keyboard_full_size = make_keyboard(keyboard_rows, 3)

# Store positions of each character in actual keyboard in a dict
char_loc_dict = {}
for x in range(keyboard_char_wid):
    for y in range(keyboard_char_len):
        char_loc_dict[actual_keyboard[x][y]]=(3*x+1, 3*y+1) 

for i in range(len(matras)):
    char_loc_dict[matras[i]] = char_loc_dict[(row_1+['ऑ'])[i+1]]



loc_char_dict = {}
for char, loc in char_loc_dict.items():
    loc_char_dict[loc]=char

# Sanity check to ensure that each character has an assinged location on the keyboard
flag=0
for i in char_set:
    if not i in char_loc_dict:
        print(i+" not assigned a location!!")
        flag=1
if flag==0:
    print("All characters assinged valid locations")

# Find path of minimum jerk
def mjtg(current_x, current_y, setpoint_x, setpoint_y, num_points, move_time):
    trajectory_x = []
    trajectory_y = []
    timestep = int(move_time * num_points)
    
    for time in range(0, timestep+1):
        trajectory_x.append(
            current_x + (setpoint_x - current_x) *
            (10.0 * (time/timestep)**3
             - 15.0 * (time/timestep)**4
             + 6.0 * (time/timestep)**5))
        
        trajectory_y.append(
            current_y + (setpoint_y - current_y) *
            (10.0 * (time/timestep)**3
             - 15.0 * (time/timestep)**4
             + 6.0 * (time/timestep)**5))

    return trajectory_x, trajectory_y

word_list = [' '.join(list(i)) for i in list(data['indic'])]

hin_words = pd.DataFrame(word_list)
hin_words.columns = ['split word']

# Flattened trajectories
def flatten_list(li):
    return [j for i in li for j in i]

# To find straight line, minimum jerk trajectories

def find_noisy_trajectory(): 
    traj_x_all = []
    traj_y_all = []
    word_all = []
    traj_y_straight = []
    loc_list_all = []
    loc_list_noisy_all = []
    random_point_all = []
    
    for word in word_list:
        loc_list = []
        for char in word.split():
            loc_list.append(char_loc_dict[char])
        loc_list_all.append(loc_list)
        for sample in range(samples_per_word):
            loc_list_noisy = []
            traj_list_x = []
            traj_list_y = [] 
            random_point_list = []
            
            # Deviate touch point from centre of the key
            for i in loc_list:    
                loc_list_noisy.append((i[0]+np.random.normal(0,1/6),i[1]+np.random.normal(0,1/6)))
            loc_list_noisy_all.append(loc_list_noisy)
            
            # Minimum jerk trajectory along straight line
            for i in range(0, len(loc_list)-1):
                init = loc_list_noisy[i]
                end = loc_list_noisy[i+1]
                dist = np.sqrt((init[0]-end[0])**2 + (init[1]-end[1])**2)
                traj_tuple = mjtg(init[0],init[1],end[0],end[1],max(int(dist*0.8),1),1)
                traj_list_x.append(traj_tuple[0])
                traj_list_y.append(traj_tuple[1])
                # print(max(int(dist/2),1))
            # Perturb the straight line by locating a random point and 
            # fitting a polynomial through it and the start & end points
            traj_list_y_new = []
            for i in range(0, len(loc_list)-1):
                init = loc_list_noisy[i] 
                end = loc_list_noisy[i+1]
                random_pt = (np.random.normal((init[0]+end[0])/2,(np.abs(init[0]-end[0]))/6), 
                                np.random.normal((init[1]+end[1])/2,(np.abs(init[1]-end[1]))/6))

                z=np.polyfit([init[0],random_pt[0],end[0]],[init[1],random_pt[1],end[1]],2)
                traj_list_y_new.append(list(np.polyval(z,traj_list_x[i])))
                random_point_list.append(random_pt)
                
            # Flatten the lists
            traj_x_all.append(flatten_list(traj_list_x))
            traj_y_all.append(flatten_list(traj_list_y_new)) # Perturbed y coordinates
            word_all.append(word)
            traj_y_straight.append(flatten_list(traj_list_y))
            random_point_all.append(random_point_list)

    return traj_x_all, traj_y_all, word_all, traj_y_straight, loc_list_all, loc_list_noisy_all, random_point_all

traj_x_all, traj_y_all, word_all, traj_y_straight, loc_list_all, loc_list_noisy_all, random_point_all = find_noisy_trajectory()

def plot_word_swipe(traj_x_all, traj_y_all, word_all, traj_y_straight, loc_list_all, loc_list_noisy_all, random_point_all, start_idx):
    # traj_y is the noisy y-coordinate list
    print(word_all[start_idx])
    
    fig = plt.figure(figsize= (15,5))
    for plot in range(0,3):
        fig.add_subplot(1,3,plot+1)
        idx = start_idx+plot
        plt.plot(traj_y_straight[idx], -np.array(traj_x_all[idx]))
        plt.plot(traj_y_all[idx], -np.array(traj_x_all[idx]),'bo')
        plt.plot(traj_y_all[idx], -np.array(traj_x_all[idx]))
        

        char_list = word_all[idx].split(' ')
        for i, txt in enumerate(char_list):
            plt.annotate(txt, (loc_list_all[idx][i][1], loc_list_all[idx][i][0])) 

        plt.title("Swipe trajectory"+str(plot+1))
        plt.legend(['straight', 'perturbed'])
        plt.grid(True)

plot_word_swipe(traj_x_all, traj_y_all, word_all, traj_y_straight, loc_list_all, loc_list_noisy_all, random_point_all, 10) 
# start_idx has to be multiple of samples_per_word

def make_one_hot(selected_idx, num_characters_on_keyboard):
    li = [0]*num_characters_on_keyboard
    li[selected_idx] = 1
    return li

def make_embedding_of_one_word(traj_x, traj_y):
    word_embed = []
    for i in range(len(traj_x)):
        if i==0:
            x_derivative=0
            y_derivative=0
        elif i==len(traj_x)-1:
            x_derivative = x_derivative # Maintain same derivative value as earlier
            y_derivative = y_derivative
        else:
            x_derivative = traj_x[i+1]-traj_x[i-1]
            y_derivative = traj_y[i+1]-traj_y[i-1]
        
        x_coord = np.clip(int(np.round(traj_x[i])),0,14)
        y_coord = np.clip(int(np.round(traj_y[i])),0,29)
        char_idx = valid_chars_dict[keyboard_full_size[x_coord][y_coord]]
        word_embed.append([traj_x[i], traj_y[i], x_derivative, y_derivative, char_idx])
        
    return word_embed

embeddings_all = []
for i in range(len(traj_x_all)):
    embeddings_all.append(make_embedding_of_one_word(traj_x_all[i], traj_y_all[i]))
    #if (i%5000==0):
    #    print(i)

training_dataset = pd.DataFrame(list(zip(word_all,embeddings_all)), columns=['word','embedding'])

#print("Length of training dataset before restricting embedding length = ", len(training_dataset))

import ast

MAX_SPAN_LENGTH = 120 # Decide based on maximum value of maxlen column
training_dataset['maxlen']=training_dataset['embedding'].apply(lambda x:len(x)) 
training_dataset = training_dataset[training_dataset['maxlen']<=MAX_SPAN_LENGTH-5] # +5 is only to a have a few <e>'s at the end of all sequences
print("Length of training dataset after restricting embedding length = ", len(training_dataset))

training_dataset.head()

training_dataset.to_csv(path+lang+'/gesture_embeddings.csv')

