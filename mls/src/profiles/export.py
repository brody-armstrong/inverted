import json, pandas as pd, numpy as np
S=2026
d=pd.read_csv(f'out/players_{S}.csv'); comps=json.load(open(f'out/comps_{S}.json')); roles=json.load(open(f'out/roles_{S}.json'))
BARS=[("xg_96","xG"),("shots_96","Shots"),("xa_96","xA"),("kp_96","Key passes"),("passes_96","Passes"),("recv_96","Receptions"),
      ("drib_96","Dribble actions"),("int_96","Interruptions"),("avg_vertical_distance_yds","Pass verticality"),("share_team_touches","Touch share"),
      ("ga_Shooting_96","g+ Shooting"),("ga_Passing_96","g+ Passing"),("ga_Receiving_96","g+ Receiving"),("ga_Dribbling_96","g+ Dribbling"),
      ("ga_Interrupting_96","g+ Interrupting"),("passes_completed_over_expected_p100","Pass acc. vs xPass")]
idx={pid:i for i,pid in enumerate(d.player_id)}
P=[]
for _,r in d.iterrows():
    nn=lambda v,k=3: None if pd.isna(v) else round(float(v),k)
    P.append({"id":r.player_id,"n":r.player_name,"t":r.team_abbreviation,"tn":r.team_short_name,"p":r.general_position,"g":r.group,"r":r.role,
      "a":nn(r.age,1),"m":int(r.minutes_played),"s":None if pd.isna(r.guaranteed_compensation) else int(r.guaranteed_compensation),
      "nat":r.nationality if isinstance(r.nationality,str) else "","tr":bool(r.traded),
      "gp":nn(r.gplus_96_adj),"gpr":nn(r.gplus_96),"ge":nn(r.gplus_expected_96),"v":nn(r.value_season,2),
      "xg":nn(r.xg_96,2),"xa":nn(r.xa_96,2),"gls":int(r.goals),"ast":int(r.primary_assists),
      "pc":[None if pd.isna(r[f"pct_{c}"]) else int(r[f"pct_{c}"]) for c,_ in BARS],
      "c":[[idx[a],s] for a,s in comps[r.player_id]]})
R={}
for k,v in roles.items():
    g,_=k.split("|"); R.setdefault(g,[]).append({"name":v["name"],"n":v["n"],"high":v["high"],"low":v["low"],"ex":v["members"][:4]})
json.dump({"season":S,"bars":[b for _,b in BARS],"players":P,"roles":R},open('out/data.json','w'),separators=(',',':'))
print(len(P), round(len(open('out/data.json').read())/1024),'KB')
