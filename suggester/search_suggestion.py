#%%  Imports
import json
import pandas as pd

#%%  Setup routes

with open("data/routes_data.json", "r") as f:
    routes_data = json.load(f)

routes = pd.DataFrame(routes_data["routes"])
routes.style

#%%  Setup sessions

with open("data/sessions_data.json", "r") as f:
    sessions_data = json.load(f)

sessions = pd.DataFrame(sessions_data["sessions"])

sessions["intention_type"] = sessions["intention"].apply(lambda x: x["type"])
sessions["intention_context"] = sessions["intention"].apply(lambda x: x["context"])
sessions = sessions.drop(columns=["intention"])
sessions.style
