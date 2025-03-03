readFile.py
import pandas as pd


# read excel file
file_path = "" #enter the file later
df = pd.read_excel(file_path, engine="openpyxl", skiprows=2) #skip the first two since they are empty

# Clean the data (delete empty rows)
df.columns = df.iloc[0]  # set the collumns number
df = df[1:].reset_index(drop=True)  #get the data

#drop the NAN collumn
df = df.loc[:, ~df.columns.isna()]
data_dict = df.to_dict(orient="list")

# turn the time into actual number
def time_to_minutes(time_str):
    if isinstance(time_str, str) and "mins" in time_str:
        return int(time_str.replace(" mins", ""))
    elif isinstance(time_str, str) and ":" in time_str:
        h, m, *_ = map(int, time_str.split(":"))
        return h * 60 + m
    return time_str

df["Drive Time"] = df["Drive Time"].apply(time_to_minutes)
df["Total Direct Time for Project for Hourly Employees (Including Drive Time)"] = df["Total Direct Time for Project for Hourly Employees (Including Drive Time)"].apply(time_to_minutes)
