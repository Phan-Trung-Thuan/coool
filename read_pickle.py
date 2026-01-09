import pandas as pd

# file_path = 'coool/baseline/video_Driver_State_Changed.pkl'
# file_path = 'coool/baseline/video_track_id.pkl'
file_path = 'coool/baseline/hazard_name_by_id_blip2flan.pkl'

df = pd.read_pickle(file_path)
print(df['video_0127'])
