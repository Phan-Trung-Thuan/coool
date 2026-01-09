import pandas as pd
import pickle

# file_path = 'coool/baseline/video_Driver_State_Changed.pkl'
# file_path = 'coool/baseline/video_track_id.pkl'
# file_path_1 = 'coool/v1/hazard_name_by_id_1.pkl'
# file_path_2 = 'coool/v1/hazard_name_by_id_2.pkl'

# df1 = pd.read_pickle(file_path_1)
# df2 = pd.read_pickle(file_path_2)
# df = df1 | df2

# print(len(df1), len(df2), len(df))

# with open("hazard_name_by_id.pkl", "wb") as f:
#     pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

file_path = 'coool/v1/hazard_name_by_id.pkl'
# file_path = 'coool/baseline/hazard_name_by_id_blip2opt.pkl'
df = pd.read_pickle(file_path)
print(df['video_0004'].keys())
