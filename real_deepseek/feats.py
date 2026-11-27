import json, glob, os, numpy as np, pandas as pd


def load(folder="/dataset"):
    """
    Read all veh_*.json files.  Each file = messages received by one vehicle.
    Each message contains sender + receiver info + `attacker` flag (per-message).
    """
    rows = []
    files = sorted(glob.glob(os.path.join(folder, "**", "veh_*.json"), recursive=True))
    if not files:
        raise FileNotFoundError("No veh_*.json found under %s" % folder)

    for f in files:
        recv = os.path.splitext(os.path.basename(f))[0]
        with open(f, encoding="utf-8") as fh:
            msgs = json.load(fh)
        for m in msgs:
            s, r = m["sender"], m["receiver"]
            sp = np.array(s["pos"].split(","), float)[:2]
            rp = np.array(r["pos"].split(","), float)[:2]
            sn = np.array(s["pos_noise"].split(","), float)[:2]
            rn = np.array(r["pos_noise"].split(","), float)[:2]
            d = sp - rp

            rows.append(dict(
                recv=recv,
                send=m["sender_id"],
                att=int(m.get("attacker", 0)),
                t=int(m["rcvTime"]),
                dt=m["rcvTime"] - m["sendTime"],
                dist=np.hypot(*d),
                dx=d[0], dy=d[1],
                road=s.get("distance_to_road_edge", 0.0),
                s_spd=s["spd"], r_spd=r["spd"],
                dspd=s["spd"] - r["spd"],
                s_acl=s["acl"],
                s_hed=s["hed"],
                dhed=((s["hed"] - r["hed"] + 180) % 360) - 180,
                s_noise=np.hypot(*sn),
                r_noise=np.hypot(*rn),
                s_spdn=s["spd_noise"],
                s_hedn=s["hed_noise"],
                s_aclnoise=s["acl_noise"],
                bearing=np.degrees(np.arctan2(d[1], d[0])),
                hed_vs_bearing=((s["hed"] - np.degrees(np.arctan2(d[1], d[0])) + 180) % 360) - 180,
                # additional BSM-observable features that help distinguish attack patterns
                s_speed_res=s["spd_noise"],                # sender-reported speed error proxy
                s_pos_res=np.hypot(*sn),                   # sender-reported position noise magnitude
                hed_diff_abs=abs(((s["hed"] - r["hed"] + 180) % 360) - 180),
            ))
    df = pd.DataFrame(rows).sort_values("t", kind="stable").reset_index(drop=True)
    return df


FEATS = [
    "dist", "road", "s_spd", "dspd", "s_acl", "dhed", "s_noise", "s_spdn",
    "s_hedn", "s_aclnoise", "hed_vs_bearing", "dt",
    "s_speed_res", "s_pos_res", "hed_diff_abs",
]