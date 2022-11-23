import pandas as pd

def DataLoader(input_data,backtrack_coefficient):
 
    df = pd.DataFrame()
    for x in range(len(input_data)):
        df_annotation_list=pd.json_normalize(input_data[x])
        for y in range(backtrack_coefficient,0,-1):
            past_speed_columns = ('OwnSpeed_tminus'+str(backtrack_coefficient-y)+'_frame')
            past_distance_columns = ('Distance_ref_tminus'+str(backtrack_coefficient-y)+'_frame')
            if(df.shape[0]>=int(y)):
            
                df_annotation_list[past_speed_columns] = df.iloc[df.shape[0]-int(y)]['OwnSpeed']
                df_annotation_list[past_distance_columns] = df.iloc[df.shape[0]-int(y)]['Distance_ref']
            else:
                df_annotation_list[past_speed_columns]=0
                df_annotation_list[past_distance_columns]=0
    
        df=pd.concat([df,df_annotation_list])
    return df