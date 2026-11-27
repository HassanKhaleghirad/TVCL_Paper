import json, glob, os, numpy as np, pandas as pd

def load(folder="/dataset"):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "**", "veh_*.json"), recursive=True)):
        recv = os.path.splitext(os.path.basename(f))[0]
        for m in json.load(open(f, encoding="utf-8")):
            s, r = m["sender"], m["receiver"]
            sp = np.array(s["pos"].split(","), float)[:2]; rp = np.array(r["pos"].split(","), float)[:2]
            sn = np.array(s["pos_noise"].split(","), float)[:2]; rn = np.array(r["pos_noise"].split(","), float)[:2]
            d = sp - rp
            rows.append(dict(recv=recv, send=m["sender_id"], att=int(m["attacker"]), t=int(m["rcvTime"]),
                dt=m["rcvTime"] - m["sendTime"], dist=np.hypot(*d), dx=d[0], dy=d[1],
                road=s.get("distance_to_road_edge", 0.0), s_spd=s["spd"], r_spd=r["spd"], dspd=s["spd"] - r["spd"],
                s_acl=s["acl"], s_hed=s["hed"], dhed=((s["hed"] - r["hed"] + 180) % 360) - 180,
                s_noise=np.hypot(*sn), r_noise=np.hypot(*rn), s_spdn=s["spd_noise"], s_hedn=s["hed_noise"],
                s_aclnoise=s["acl_noise"], bearing=np.degrees(np.arctan2(d[1], d[0])),
                hed_vs_bearing=((s["hed"] - np.degrees(np.arctan2(d[1], d[0])) + 180) % 360) - 180))
    return pd.DataFrame(rows).sort_values("t", kind="stable").reset_index(drop=True)
FEATS = ["dist","road","s_spd","dspd","s_acl","dhed","s_noise","s_spdn","s_hedn","s_aclnoise","hed_vs_bearing","dt"]
