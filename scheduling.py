import pandas as pd


# read excel file
file_path = "Data.xlsx" #enter the file later›
df = pd.read_excel(file_path, engine="openpyxl", skiprows=1) #skip the first two since they are empty

# Clean the data (delete empty rows)
df.columns = df.iloc[0]  # set the collumns number
df = df[1:].reset_index(drop=True)  #get the data

#drop the NAN collumn
df = df.loc[:, ~df.columns.isna()]
data_dict = df.to_dict(orient="list")

# turn the time into actual number
def time_to_minutes(time_str):
    if isinstance(time_str, str):
        time_str = time_str.lower().strip()  

        if "hour" in time_str or "h" in time_str:  
            time_str = time_str.replace("hours", "").replace("hour", "").replace("h", "").strip()
            parts = time_str.replace("mins", "").replace("min", "").replace("m", "").strip().split()

            if len(parts) == 2:  # "1 hour 32 mins" -> ["1", "32"]
                return int(parts[0]) * 60 + int(parts[1])  
            return int(parts[0]) * 60  # "1 hour" -> ["1"]

        elif "min" in time_str or "m" in time_str:  
            return int(time_str.replace("mins", "").replace("min", "").replace("m", "").strip())

        elif ":" in time_str:  
            h, m, *_ = map(int, time_str.split(":"))
            return h * 60 + m

    return None  



df["Drive Time"] = df["Drive Time"].apply(time_to_minutes)
df["Total Direct Time for Project for Hourly Employees (Including Drive Time)"] = df["Total Direct Time for Project for Hourly Employees (Including Drive Time)"].apply(time_to_minutes)

print (df)