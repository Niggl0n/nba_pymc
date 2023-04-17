import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import pandas as pd
import xarray as xr
from sklearn.preprocessing import scale
import scipy
from scipy.special import expit
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt

class TeamLabeller(az.labels.BaseLabeller):
    def make_label_flat(self, var_name, sel, isel):
        sel_str = self.sel_to_str(sel, isel)
        return sel_str


# df_nugget_box = pd.read_csv("data/df_nuggets_games.csv", index_col="Unnamed: 0")
# df_nugget_box["is_win"] = np.where(df_nugget_box["WL"]=="W", 1, 0)
# df_nugget_box["AST_PCT"] = df_nugget_box["AST"] / df_nugget_box["FGM"]
# df_nugget_box["FG3_PCT_OF_FG_TOT"] = df_nugget_box["FG3A"] / df_nugget_box["FGA"]
# df_nugget_box.head()

df_all_games = pd.read_csv("data/gamelogs_allteams_22_23.csv", index_col=0)
df_all_games["team_name"] = df_all_games["MATCHUP"].str.split(" ").str[0] 
df_all_games["is_win"] = np.where(df_all_games["WL"]=="W", 1, 0)
df_all_games["is_home"] = np.where(df_all_games["MATCHUP"].str.contains("vs."), 1, 0)
df_all_games["AST_PCT"] = df_all_games["AST"] / df_all_games["FGM"]
df_all_games["FG3_PCT_OF_FG_TOT"] = df_all_games["FG3A"] / df_all_games["FGA"]
df_all_games["GAME_DATE"] = pd.to_datetime(df_all_games["GAME_DATE"])

# add column wich contains number of days between games for each team
df_all_games = df_all_games.sort_values(["team_name", "GAME_DATE"], ascending=[True, True])
df_all_games["days_since_last_game"] = df_all_games.groupby("team_name")["GAME_DATE"].diff().dt.days.abs()
# add columns which contains number of games played in the last 7 days per team

# set GAME_DATE as index
df_all_games = df_all_games.set_index("GAME_DATE")

df_all_games['games_played_last_7_days'] = df_all_games.groupby('team_name')['TEAM_ID'].rolling('7D').count() # .reset_index(0, drop=True)
df_all_games.head(5)


team_idx, teams = pd.factorize(df_all_games["team_name"], sort=True)
# away_idx, _ = pd.factorize(df_all_games["away_team"], sort=True)
coords = {"team": teams}

# pymc model to predict points scored
with pm.Model(coords=coords) as home_away_allteams:
    team = pm.ConstantData("home_team", team_idx, dim="games")
    home_away = df_all_games["is_home"].astype("category").cat.codes
    
    team_strength = pm.Normal("mu",100, 30, dims="team")
    mu = team_strength[team_idx]
    points_scored = pm.Poisson("points_scored", mu=mu, observed=df_all_games["PTS"])
    trace = pm.sample(draws=1000,tune=1500)
    
az.plot_trace(trace, var_names=["mu"])
az.summary(trace, kind="diagnostics")
az.plot_forest(trace, var_names=["mu"], coords={"team": teams}, combined=True, figsize=(10, 5))


# The above code defines a home advantage model for the NBA data. It is a 
# hierarchical model that includes a team strength model and a home advantage
# model. The team strength model is a normal distribution with mean 100 and 
# standard deviation 30. The home advantage model is a normal distribution with
# mean 0 and standard deviation 10. The team strength and home advantage 
# distributions are then used to generate a points scored variable.
with pm.Model(coords=coords) as home_adv_model:
    team = pm.ConstantData("home_team", team_idx, dim="games")
    is_home = df_all_games["is_home"].values
    # team strength model
    team_strength = pm.Normal("mu", 100, 30, dims="team")
    mu = team_strength[team_idx]
    # home advantage model
    home_adv = pm.Normal("home_advantage", 0, 10, dims="team")
    home_adv_vector = pm.math.switch(is_home, home_adv[team_idx], -home_adv[team_idx])
    mu += home_adv_vector
    # likelihood
    points_scored = pm.Poisson("points_scored", mu=mu, observed=df_all_games["PTS"])
    trace = pm.sample(draws=1000, tune=1500)
    
# alternative model
# with pm.Model(coords=coords) as home_away_allteams:
#     team_id = pm.ConstantData("team_id", team_idx, dim="games")
#     is_home = pm.ConstantData("is_home", is_home_idx, dims="games")
#     
#     
#     base_score = pm.Normal("base_score", 100, 20)
#     sd_strength = pm.HalfNormal("sd_strength", sigma=5)
#     sd_home = pm.HalfNormal("sd_home", sigma=5)
#     team_strength = pm.Normal("team_strength", 0, sd_strength, dims="team")
#     home_advantage = pm.Normal("home_advantage", 0, sd_home, dims="team")
#     mu = base_score + team_strength[team_idx] + home_advantage[team_id]*is_home

with home_away_allteams:
    home_adv_trace = trace.posterior["home_advantage"]
    
az.plot_forest(home_adv_trace, var_names="home_advantage", coords={"team": teams})
ax[0].set_title("home_advantage");
plt.title("Comparison of Home Advantage between Model 1 and Model 2")
plt.show()
# az.plot_forest([trace, trace], model_names=["Model 1", "Model 2"], var_names=["home_advantage"])



