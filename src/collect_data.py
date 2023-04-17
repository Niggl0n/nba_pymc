import pandas as pd
import numpy as np
import json
import difflib
import time
import requests

from nba_api.stats.endpoints import leaguestandingsv3
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.endpoints import boxscoretraditionalv2


def get_team_ids():
    box = leaguestandingsv3.LeagueStandingsV3()
    df = box.get_data_frames()[0]
    return dict(zip(df.TeamName,df.TeamID))


def get_team_gamelog(team_id):
    games = teamgamelog.TeamGameLog(team_id=team_id)
    return games.get_data_frames()[0]

def get_box_scores(game_id):
    box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id)
    dfs = box.get_data_frames()
    return dfs


teams_dict = get_team_ids()
nuggets_id = teams_dict["Nuggets"]
df_games_nuggets = get_team_gamelog(nuggets_id)
df_games_nuggets["is_home"] = np.where(df_games_nuggets["MATCHUP"].str.contains("vs."), True, False)
dfs_player_bs_nuggets = []
dfs_team_bs_nuggets = []
dfs_starter_bs_nuggets = []
for i,game_id in enumerate(df_games_nuggets["Game_ID"].unique()):
    # if i == 3:
    #     break
    print(game_id)
    time.sleep(0.6)
    df_player_bs_nuggets, df_team_bs_nuggets, df_starter_bs_nuggets = get_box_scores(game_id)
    dfs_player_bs_nuggets.append(df_player_bs_nuggets)
    dfs_team_bs_nuggets.append(df_team_bs_nuggets)
    dfs_starter_bs_nuggets.append(df_starter_bs_nuggets)

dfs_player_bs_nuggets = pd.concat(dfs_player_bs_nuggets,axis=0)
dfs_team_bs_nuggets = pd.concat(dfs_team_bs_nuggets,axis=0)
dfs_starter_bs_nuggets = pd.concat(dfs_starter_bs_nuggets,axis=0)

dfs_player_bs_nuggets.to_csv("data/dfs_player_box_nuggets.csv")
dfs_team_bs_nuggets.to_csv("data/dfs_team_box_nuggets.csv")
dfs_starter_bs_nuggets.to_csv("data/dfs_starter_box_nuggets.csv")
df_games_nuggets.to_csv("data/df_nuggets_games.csv")
print("")

