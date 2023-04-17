import pandas as pd
import numpy as np
import json
import difflib
import time
import requests


from nba_api.stats.endpoints import defensehub


# Retry Wrapper
def retry(func, retries=3):
    def retry_wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                print(e)
                time.sleep(30)
                attempts += 1

    return retry_wrapper



# game_id="0022201055"
# box = defensehub.DefenseHub(game_scope_detailed="Season", league_id="00", player_or_team="Team", player_scope="All Players", season_type_playoffs="Regular Season", season="2022-23")
@retry
def get_def():
    return defensehub.DefenseHub()

box = get_def()
dfs = box.get_data_frames()

for df in dfs:
    print("\n")
    print(df.shape)

print("dsd")