"""Hasta (mudra) specifications.

Each single-hand (asamyuta) hasta is described by
  * an *intent* pose: approximate joint angles in degrees (from the
    Abhinaya Darpana verbal descriptions), and
  * explicit contact constraints: geom pairs that must touch ("contacts")
    or must stay apart ("apart").
targets.py turns the intent into a physically valid target by solving for
the nearest configuration that satisfies the contacts, then settling it in
the simulator.  NOTE: these are author-specified from textual descriptions
and are NOT yet validated by a Bharatanatyam expert (see README).
"""
from __future__ import annotations

E = dict(mcp=0, pip=0, dip=0)           # extended
F = dict(mcp=88, pip=100, dip=60)       # fully flexed into the palm
C = dict(mcp=30, pip=40, dip=30)        # curved
B = dict(mcp=75, pip=85, dip=45)        # bent (as in Tripataka ring finger)
S = dict(mcp=25, pip=12, dip=8)         # slight bend (Sarpashirsha)
H = dict(mcp=10, pip=95, dip=70)        # hooked (Tamrachuda index)

T_ADD = dict(cmc_flex=0, cmc_abd=-40, cmc_rot=0, mcp=5, ip=5)   # thumb bent alongside index
T_OUT = dict(cmc_flex=-20, cmc_abd=30, cmc_rot=0, mcp=0, ip=0)      # stretched away (Ardhachandra)
T_UP = dict(cmc_flex=-25, cmc_abd=-30, cmc_rot=0, mcp=0, ip=-10)     # erect (Shikhara)
T_OVER = dict(cmc_flex=45, cmc_abd=-10, cmc_rot=10, mcp=30, ip=20)  # over flexed fingers (Mushti)
T_OPP = dict(cmc_flex=55, cmc_abd=10, cmc_rot=20, mcp=20, ip=15)    # opposed (tip-to-tip)
T_CURVE = dict(cmc_flex=25, cmc_abd=15, cmc_rot=10, mcp=15, ip=20)  # cupped


def fing(index=E, middle=E, ring=E, little=E, thumb=T_ADD, spread=(0, 0, 0, 0)):
    q = {}
    for name, d, a in zip(["index", "middle", "ring", "little"], [index, middle, ring, little], spread):
        q[f"{name}_abd"] = a
        for k, v in d.items():
            q[f"{name}_{k}"] = v
    for k, v in thumb.items():
        q[f"thumb_{k}"] = v
    return q


def tip(f):
    return f"{f}_dist"


TOGETHER = (-5, 0, 6, 8)
SPREAD = (8, 0, -5, -10)
WIDE = (16, 4, -8, -16)

# name -> (meaning, intent dict, contacts [(g1,g2)], apart [(g1,g2)])
HASTAS = {
    "pataka": ("flag", fing(spread=TOGETHER), [("thumb_dist", "index_prox")], []),
    "tripataka": ("three parts of a flag", fing(ring=B, spread=TOGETHER), [("thumb_dist", "index_prox")], []),
    "ardhapataka": ("half flag", fing(ring=B, little=B, spread=TOGETHER), [("thumb_dist", "index_prox")], []),
    "kartarimukha": ("scissors face", fing(ring=F, little=F, thumb=T_OVER, spread=(14, -8, 0, 0)),
                     [("thumb_dist", "ring_mid")], [("index_dist", "middle_dist")]),
    "mayura": ("peacock", fing(ring=dict(mcp=55, pip=60, dip=25), thumb=T_OPP, spread=(6, 0, 0, -12)),
               [("thumb_dist", "ring_dist")], []),
    "ardhachandra": ("half moon", fing(thumb=T_OUT, spread=TOGETHER), [], [("thumb_dist", "index_prox"), ("thumb_dist", "index_mid")]),
    "arala": ("bent", fing(index=dict(mcp=35, pip=60, dip=45), spread=TOGETHER), [], []),
    "shukatunda": ("parrot's beak", fing(index=dict(mcp=35, pip=60, dip=45), ring=B, spread=TOGETHER), [], []),
    "mushti": ("fist", fing(F, F, F, F, thumb=T_OVER), [("thumb_dist", "middle_mid")], []),
    "shikhara": ("peak", fing(F, F, F, F, thumb=T_UP), [], [("thumb_dist", "index_mid"), ("thumb_dist", "middle_mid")]),
    "kapittha": ("wood-apple", fing(dict(mcp=60, pip=80, dip=50), F, F, F, thumb=dict(cmc_flex=15, cmc_abd=-15, cmc_rot=10, mcp=10, ip=5)),
                 [("index_dist", "thumb_dist")], []),
    "katakamukha": ("opening in a bracelet", fing(dict(mcp=50, pip=40, dip=20), dict(mcp=50, pip=45, dip=20), S, S, thumb=T_OPP, spread=(4, 0, -6, -12)),
                    [("thumb_dist", "index_dist"), ("thumb_dist", "middle_dist")], []),
    "suchi": ("needle", fing(E, F, F, F, thumb=T_OVER), [("thumb_dist", "middle_mid")], []),
    "chandrakala": ("digit of the moon", fing(E, F, F, F, thumb=T_OUT), [], [("thumb_dist", "middle_mid"), ("thumb_dist", "index_prox")]),
    "padmakosha": ("lotus bud", fing(C, C, C, C, thumb=T_CURVE, spread=SPREAD), [], [("thumb_dist", "index_dist")]),
    "sarpashirsha": ("snake hood", fing(S, S, S, S, spread=TOGETHER), [("thumb_dist", "index_prox")], []),
    "mrigashirsha": ("deer head", fing(S, S, S, dict(mcp=-10, pip=0, dip=0), thumb=T_UP, spread=(-3, 0, 2, -20)), [], []),
    "simhamukha": ("lion face", fing(E, dict(mcp=55, pip=55, dip=25), dict(mcp=55, pip=55, dip=25), E, thumb=T_OPP, spread=(10, 0, 0, -18)),
                   [("thumb_dist", "middle_dist"), ("thumb_dist", "ring_dist")], []),
    "kangula": ("bell", fing(C, C, F, C, thumb=T_CURVE, spread=SPREAD), [], []),
    "alapadma": ("full-blown lotus", fing(dict(mcp=-5, pip=0, dip=0), dict(mcp=5, pip=5, dip=0), dict(mcp=15, pip=10, dip=5), dict(mcp=30, pip=20, dip=10), thumb=T_OUT, spread=WIDE), [], []),
    "chatura": ("clever", fing(E, E, E, dict(mcp=-5, pip=0, dip=0), thumb=dict(cmc_flex=55, cmc_abd=-5, cmc_rot=10, mcp=30, ip=20), spread=(-3, 0, 2, -22)),
                [("thumb_dist", "middle_prox")], []),
    "bhramara": ("bee", fing(dict(mcp=60, pip=70, dip=40), dict(mcp=45, pip=45, dip=20), E, E, thumb=T_OPP, spread=(0, 0, -6, -14)),
                 [("thumb_dist", "middle_dist")], []),
    "hamsasya": ("swan's beak", fing(dict(mcp=40, pip=35, dip=15), E, E, E, thumb=T_OPP, spread=(4, 0, -10, -18)),
                 [("thumb_dist", "index_dist")], []),
    "hamsapaksha": ("swan's wing", fing(S, S, S, dict(mcp=-5, pip=0, dip=0), thumb=T_ADD, spread=(-3, 0, 2, -22)), [], []),
    "sandamsha": ("tongs", fing(dict(mcp=45, pip=40, dip=20), C, C, C, thumb=T_OPP, spread=(4, 0, -6, -10)),
                  [("thumb_dist", "index_dist")], [("thumb_dist", "middle_dist")]),
    "mukula": ("bud", fing(dict(mcp=45, pip=35, dip=15), dict(mcp=50, pip=40, dip=15), dict(mcp=55, pip=40, dip=15), dict(mcp=60, pip=40, dip=15), thumb=T_OPP, spread=(-8, -2, 4, 10)),
               [("thumb_dist", "index_dist"), ("thumb_dist", "middle_dist"), ("middle_dist", "ring_dist"), ("ring_dist", "little_dist")], []),
    "tamrachuda": ("rooster", fing(H, F, F, F, thumb=T_OVER), [("thumb_dist", "middle_mid")], []),
    "trishula": ("trident", fing(E, E, E, dict(mcp=65, pip=60, dip=30), thumb=T_OPP, spread=(12, 0, -10, 0)),
                 [("thumb_dist", "little_dist")], []),
}

CORE8 = ["pataka", "tripataka", "ardhachandra", "kartarimukha", "mayura", "mushti", "shikhara", "suchi"]
ALL28 = list(HASTAS)
assert len(ALL28) == 28

# Display orientation (root roll/pitch/yaw in degrees).  Most hastas are
# shown palm-to-audience, fingers up; a few are conventionally shown in
# profile or tilted.  Benchmark targets also sample a small extra rotation.
DISPLAY = {k: (0, 0, 0) for k in ALL28}
DISPLAY.update({"shikhara": (90, 0, 0), "mushti": (0, 0, 0), "kapittha": (60, 0, 0), "suchi": (30, 0, 0),
                "tamrachuda": (60, 0, 0), "katakamukha": (45, 0, 0), "mukula": (60, -20, 0),
                "hamsasya": (45, 0, 0), "sandamsha": (45, 0, 0), "bhramara": (30, 0, 0),
                "kartarimukha": (20, 0, 0), "padmakosha": (0, 20, 0), "kangula": (0, 20, 0)})


# ---------------------------------------------------------------- two-hand
# Samyuta hastas = two single-hand shapes + relative placement. Hands face the
# camera like a dancer facing the audience (right hand on image-left).  Root
# values are mirror-symmetric: (tx toward the midline, ty, tz, roll, pitch, yaw).
# Each: (meaning, right shape, left shape, right root, left root, cross contacts, cross apart, views)
SAMYUTA = {
    "anjali": ("salutation", "pataka", "pataka", (0.09, 0, 0, 90, 0, 0), (0.09, 0, 0, 90, 0, 0),
               [("R_palm", "L_palm"), ("R_middle_dist", "L_middle_dist")], [], ("front", "top")),
    "kapota": ("dove", "sarpashirsha", "sarpashirsha", (0.085, 0, 0, 90, 0, 0), (0.085, 0, 0, 90, 0, 0),
               [("R_middle_dist", "L_middle_dist"), ("R_index_dist", "L_index_dist")], [("R_palm", "L_palm")], ("front", "top")),
    "swastika": ("crossed (auspicious sign)", "pataka", "pataka", (0.11, 0, 0, 0, 0, 30), (0.11, 0.026, 0, 0, 0, 30),
                 [("R_palm", "L_palm")], [], ("front", "side")),
    "kartariswastika": ("crossed scissors", "kartarimukha", "kartarimukha", (0.11, 0, 0, 0, 0, 30), (0.11, 0.026, 0, 0, 0, 30),
                        [("R_palm", "L_palm")], [], ("front", "side")),
    "pushpaputa": ("flower casket", "sarpashirsha", "sarpashirsha", (0.065, 0, 0, 180, -90, 0), (0.065, 0, 0, 180, -90, 0),
                   [("R_little_prox", "L_little_prox")], [], ("front", "top")),
}


# ---------------------------------------------------------------- descriptions
# Formation rules paraphrased from the Abhinaya Darpana (the "book" the agent may
# consult with read_description, at a score penalty).
DESCRIPTIONS = {
    "pataka": "Pataka (flag): all four fingers extended and held close together; the thumb is bent and pressed against the side of the index finger.",
    "tripataka": "Tripataka (three parts of a flag): as Pataka, but the ring finger is bent down.",
    "ardhapataka": "Ardhapataka (half flag): as Tripataka, with the little finger also bent; index and middle fingers stay up, thumb alongside.",
    "kartarimukha": "Kartarimukha (scissors face): index and middle fingers extended and separated like scissor blades; ring and little fingers folded into the palm, held down by the thumb.",
    "mayura": "Mayura (peacock): the tip of the ring finger touches the tip of the thumb; the other fingers are extended and slightly separated.",
    "ardhachandra": "Ardhachandra (half moon): as Pataka, but the thumb is stretched out away from the fingers, forming a crescent between thumb and index.",
    "arala": "Arala (bent): as Pataka, but the index finger is bent in a curve.",
    "shukatunda": "Shukatunda (parrot's beak): as Arala, with the ring finger also bent.",
    "mushti": "Mushti (fist): all four fingers curled tightly into the palm, the thumb pressed over the middle phalanges of the fingers.",
    "shikhara": "Shikhara (peak): a fist (Mushti) with the thumb raised straight up.",
    "kapittha": "Kapittha (wood-apple): as Shikhara, but the index finger curls over and presses onto the tip of the thumb.",
    "katakamukha": "Katakamukha (opening in a bracelet): the tips of the thumb, index and middle fingers are joined; ring and little fingers are raised and slightly curved.",
    "suchi": "Suchi (needle): the index finger points straight up; the other fingers are closed into a fist with the thumb over the middle finger.",
    "chandrakala": "Chandrakala (digit of the moon): as Suchi, but the thumb is stretched out, forming an L with the raised index finger.",
    "padmakosha": "Padmakosha (lotus bud): fingers separated and curved, thumb curved as well, as if holding a ball; fingertips do not touch.",
    "sarpashirsha": "Sarpashirsha (snake head): as Pataka, but the fingers are slightly bent at the knuckles so the palm is hollowed like a snake's hood.",
    "mrigashirsha": "Mrigashirsha (deer head): as Sarpashirsha, with the thumb and the little finger raised up.",
    "simhamukha": "Simhamukha (lion face): the tips of the middle and ring fingers touch the thumb; index and little fingers are extended.",
    "kangula": "Kangula (bell): as Padmakosha, but the ring finger is bent down into the palm.",
    "alapadma": "Alapadma (full-blown lotus): all fingers spread wide apart like a fan, each successive finger curving a little more toward the little finger side; thumb stretched out.",
    "chatura": "Chatura (clever): index, middle and ring fingers extended together, the little finger extended apart; the thumb is placed at the base of the middle finger.",
    "bhramara": "Bhramara (bee): the middle finger tip touches the thumb tip, the index finger is curled, ring and little fingers are extended.",
    "hamsasya": "Hamsasya (swan's beak): the tips of the thumb and index finger touch; the other three fingers are extended and separated.",
    "hamsapaksha": "Hamsapaksha (swan's wing): as Sarpashirsha, with the little finger stretched out to the side.",
    "sandamsha": "Sandamsha (tongs): the index finger and thumb pinch together like tongs, the other fingers curved and apart.",
    "mukula": "Mukula (bud): all five fingertips brought together at one point, like a closed bud.",
    "tamrachuda": "Tamrachuda (rooster): a fist with the index finger raised and hooked, the thumb pressing on the middle finger.",
    "trishula": "Trishula (trident): the thumb and little finger touch (little finger bent); index, middle and ring fingers stand up separated like a trident.",
    # samyuta
    "anjali": "Anjali (salutation): both hands in Pataka, palms pressed together, fingers pointing up.",
    "kapota": "Kapota (dove): as Anjali, but the palms are hollowed (Sarpashirsha) so only the fingertips and the edges of the hands meet.",
    "swastika": "Swastika (auspicious cross): two Pataka hands crossed at the wrists, palms facing out, fingers spread apart above the crossing.",
    "kartariswastika": "Kartariswastika (crossed scissors): two Kartarimukha hands crossed at the wrists.",
    "pushpaputa": "Pushpaputa (flower casket): two Sarpashirsha hands held palms up side by side, little-finger edges touching, as if offering flowers.",
}
