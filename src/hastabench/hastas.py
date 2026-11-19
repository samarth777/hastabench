"""Hasta (mudra) specifications.

Each single-hand (asamyuta) hasta is described by
  * an *intent* pose: approximate joint angles in degrees (from the
    Abhinaya Darpana verbal descriptions), and
  * explicit contact constraints: geom pairs that must touch ("contacts")
    or must stay apart ("apart").
targets.py turns the intent into a physically valid target by solving for
the nearest configuration that satisfies the contacts, then settling it in
the simulator.  v1.1: every target was checked against >= 4 reference images
(verify/REPORT.md) and corrected; still not reviewed by a Bharatanatyam expert.
"""
from __future__ import annotations

E = dict(mcp=0, pip=0, dip=0)           # extended
F = dict(mcp=88, pip=100, dip=60)       # fully flexed into the palm
C = dict(mcp=30, pip=40, dip=30)        # curved
# v1.1 corrections from the reference-image verification (verify/REPORT.md):
B = dict(mcp=35, pip=95, dip=50)        # bent at the middle joint, tip pointing forward (Tripataka ring)
S = dict(mcp=25, pip=30, dip=20)        # hooded (Sarpashirsha)

T_ADD = dict(cmc_flex=0, cmc_abd=-40, cmc_rot=0, mcp=12, ip=25)     # thumb bent, pressed alongside the index
T_OUT = dict(cmc_flex=-20, cmc_abd=30, cmc_rot=0, mcp=0, ip=0)      # stretched away (Ardhachandra)
T_UP = dict(cmc_flex=-25, cmc_abd=-15, cmc_rot=0, mcp=0, ip=-10)    # erect (Shikhara)
T_OVER = dict(cmc_flex=40, cmc_abd=-15, cmc_rot=25, mcp=50, ip=35)  # lying flat across the folded fingers (Mushti)
T_OPP = dict(cmc_flex=55, cmc_abd=10, cmc_rot=20, mcp=20, ip=15)    # opposed (tip-to-tip)
T_CUP = dict(cmc_flex=40, cmc_abd=20, cmc_rot=25, mcp=15, ip=20)    # curved up, forming the side of a cup


def fing(index=E, middle=E, ring=E, little=E, thumb=T_ADD, spread=(0, 0, 0, 0)):
    q = {}
    for name, d, a in zip(["index", "middle", "ring", "little"], [index, middle, ring, little], spread):
        q[f"{name}_abd"] = a
        for k, v in d.items():
            q[f"{name}_{k}"] = v
    for k, v in thumb.items():
        q[f"thumb_{k}"] = v
    return q


def j(mcp, pip, dip):
    return dict(mcp=mcp, pip=pip, dip=dip)


TOGETHER = (-7, 0, 9, 13)     # no gaps between the fingers
SPREAD = (10, 0, -6, -14)

# name -> (meaning, intent dict, contacts [(g1,g2)], apart [(g1,g2)])
HASTAS = {
    "pataka": ("flag", fing(spread=TOGETHER), [("thumb_dist", "index_prox")], []),
    "tripataka": ("three parts of a flag", fing(ring=B, spread=TOGETHER), [("thumb_dist", "index_prox")], []),
    "ardhapataka": ("half flag", fing(ring=B, little=j(40, 95, 50), spread=TOGETHER), [("thumb_dist", "index_prox")], []),
    "kartarimukha": ("scissors face", fing(ring=F, little=F, thumb=T_OVER, spread=(16, -10, 0, 0)),
                     [("thumb_dist", "ring_mid")], [("index_dist", "middle_dist")]),
    "mayura": ("peacock", fing(ring=j(55, 60, 25), thumb=T_OPP, spread=(-3, 0, 0, -4)),
               [("thumb_dist", "ring_dist")], []),
    "ardhachandra": ("half moon", fing(thumb=T_OUT, spread=TOGETHER), [], [("thumb_dist", "index_prox"), ("thumb_dist", "index_mid")]),
    "arala": ("bent", fing(index=j(20, 92, 55), spread=TOGETHER), [], [("thumb_dist", "index_dist")]),
    "shukatunda": ("parrot's beak", fing(index=j(45, 75, 35), ring=j(50, 80, 35), spread=TOGETHER), [], []),
    "mushti": ("fist", fing(F, F, F, F, thumb=T_OVER), [("thumb_dist", "middle_mid")], []),
    "shikhara": ("peak", fing(F, F, F, F, thumb=T_UP), [], [("thumb_dist", "index_mid"), ("thumb_dist", "middle_mid")]),
    "kapittha": ("wood-apple", fing(j(72, 90, 45), F, F, F, thumb=dict(cmc_flex=0, cmc_abd=-25, cmc_rot=0, mcp=5, ip=5)),
                 [("index_dist", "thumb_dist")], []),
    "katakamukha": ("opening in a bracelet", fing(j(50, 40, 20), j(50, 45, 20), j(15, 10, 5), j(15, 10, 5), thumb=T_OPP, spread=(4, 0, -6, -12)),
                    [("thumb_dist", "index_dist"), ("thumb_dist", "middle_dist")], []),
    "suchi": ("needle", fing(E, F, F, F, thumb=T_OVER), [("thumb_dist", "middle_mid")], []),
    "chandrakala": ("digit of the moon", fing(E, F, F, F, thumb=T_OUT), [], [("thumb_dist", "middle_mid"), ("thumb_dist", "index_prox")]),
    "padmakosha": ("lotus bud", fing(C, C, C, C, thumb=T_CUP, spread=SPREAD), [], [("thumb_dist", "index_dist")]),
    "sarpashirsha": ("snake hood", fing(S, S, S, S, spread=(-8, 0, 8, 12)), [("thumb_dist", "index_prox")], []),
    "mrigashirsha": ("deer head", fing(j(80, 30, 15), j(80, 30, 15), j(80, 30, 15), j(-5, 0, 0), thumb=dict(cmc_flex=-20, cmc_abd=-25, cmc_rot=0, mcp=0, ip=0), spread=(-3, 0, 3, -6)),
                     [], [("thumb_dist", "index_prox"), ("little_dist", "ring_dist")]),
    "simhamukha": ("lion face", fing(E, j(55, 55, 25), j(55, 55, 25), E, thumb=T_OPP, spread=(4, 0, 0, -8)),
                   [("thumb_dist", "middle_dist"), ("thumb_dist", "ring_dist")], []),
    "kangula": ("bell", fing(j(50, 35, 15), j(50, 35, 15), F, E, thumb=T_OPP, spread=(4, 0, 0, -10)),
                [("thumb_dist", "index_dist"), ("thumb_dist", "middle_dist")], [("little_dist", "ring_dist")]),
    "alapadma": ("full-blown lotus", fing(j(-5, 0, 0), j(10, 5, 0), j(30, 20, 10), j(50, 25, 10), thumb=dict(cmc_flex=-10, cmc_abd=35, cmc_rot=0, mcp=10, ip=10), spread=(20, 6, -14, -30)), [], []),
    "chatura": ("clever", fing(j(30, 10, 5), j(30, 10, 5), j(30, 10, 5), j(-5, 0, 0), thumb=dict(cmc_flex=60, cmc_abd=0, cmc_rot=25, mcp=30, ip=30), spread=(-3, 0, 3, -12)),
                [("thumb_dist", "ring_prox")], [("thumb_dist", "index_dist"), ("thumb_dist", "middle_dist")]),
    "bhramara": ("bee", fing(j(70, 85, 45), j(45, 45, 20), E, E, thumb=T_OPP, spread=(0, 0, -6, -14)),
                 [("thumb_dist", "middle_dist")], []),
    "hamsasya": ("swan's beak", fing(j(40, 35, 15), E, E, E, thumb=T_OPP, spread=(4, 0, -10, -18)),
                 [("thumb_dist", "index_dist")], []),
    "hamsapaksha": ("swan's wing", fing(j(10, 5, 0), j(10, 5, 0), j(10, 5, 0), E, thumb=dict(cmc_flex=15, cmc_abd=-40, cmc_rot=0, mcp=20, ip=10), spread=(-3, 0, 3, -38)),
                    [], [("little_dist", "ring_dist")]),
    # Abhinaya Darpana: Padmakosha opened and closed repeatedly; we use the closed frame
    "sandamsha": ("tongs", fing(j(35, 65, 40), j(40, 65, 40), j(45, 65, 40), j(50, 65, 40), thumb=T_OPP, spread=(-6, 0, 6, 12)),
                  [("thumb_dist", "index_dist"), ("thumb_dist", "middle_dist"), ("thumb_dist", "ring_dist")], []),
    "mukula": ("bud", fing(j(35, 10, 5), j(40, 10, 5), j(45, 10, 5), j(50, 15, 5), thumb=dict(cmc_flex=55, cmc_abd=0, cmc_rot=20, mcp=15, ip=5), spread=(-10, 0, 10, 20)),
               [("thumb_dist", "index_dist"), ("thumb_dist", "middle_dist"), ("thumb_dist", "ring_dist"), ("ring_dist", "little_dist")], []),
    "tamrachuda": ("rooster", fing(j(0, 70, 30), F, F, F, thumb=T_OVER), [("thumb_dist", "middle_mid")], []),
    "trishula": ("trident", fing(E, E, E, j(65, 80, 30), thumb=T_OPP, spread=(12, 0, -10, 0)),
                 [("thumb_dist", "little_dist")], []),
}

CORE8 = ["pataka", "tripataka", "ardhachandra", "kartarimukha", "mayura", "mushti", "shikhara", "suchi"]
ALL28 = list(HASTAS)
assert len(ALL28) == 28

# Display orientation (root roll/pitch/yaw in degrees).  Most hastas are
# shown palm-to-audience, fingers up; a few are conventionally shown in
# profile or tilted.
DISPLAY = {k: (0, 0, 0) for k in ALL28}
DISPLAY.update({"kapittha": (60, 0, 0), "suchi": (30, 0, 0),
                "tamrachuda": (60, 0, 0), "katakamukha": (45, 0, 0), "mukula": (20, 0, 0),
                "hamsasya": (45, 0, 0), "sandamsha": (20, 0, 0), "bhramara": (30, 0, 0),
                "kartarimukha": (20, 0, 0), "padmakosha": (0, 20, 0), "kangula": (30, 0, 0)})


# ---------------------------------------------------------------- two-hand
# Samyuta hastas = two single-hand shapes + relative placement. Hands face the
# camera like a dancer facing the audience (right hand on image-left).  Root
# values are mirror-symmetric: (tx toward the midline, ty, tz, roll, pitch, yaw).
# Each: (meaning, right shape, left shape, right root, left root, cross contacts, cross apart, views)
SAMYUTA = {
    "anjali": ("salutation", "pataka", "pataka", (0.09, 0, 0, 90, 0, 0), (0.09, 0, 0, 90, 0, 0),
               [("R_palm", "L_palm"), ("R_middle_dist", "L_middle_dist"), ("R_palm_heel", "L_palm_heel")], [], ("front", "top")),
    "kapota": ("dove", "sarpashirsha", "sarpashirsha", (0.08, 0, 0, 90, 0, 0), (0.08, 0, 0, 90, 0, 0),
               [("R_middle_dist", "L_middle_dist"), ("R_palm_heel", "L_palm_heel")], [], ("front", "top")),
    # crossed at the wrists, hands held in profile (palms toward the midline), narrow V
    "swastika": ("crossed (auspicious sign)", "pataka", "pataka", (0.11, 0, 0, -90, 0, 18), (0.11, 0.03, 0, -90, 0, 18),
                 [("R_palm_heel", "L_palm_heel")], [("R_middle_dist", "L_middle_dist")], ("front", "side")),
    "kartariswastika": ("crossed scissors", "kartarimukha", "kartarimukha", (0.11, 0, 0, -20, 0, 15), (0.11, 0.026, 0, -20, 0, 15),
                        [("R_palm_heel", "L_palm_heel")], [], ("front", "side")),
    "pushpaputa": ("flower casket", "sarpashirsha", "sarpashirsha", (0.065, 0, 0, 180, -90, 0), (0.065, 0, 0, 180, -90, 0),
                   [("R_little_prox", "L_little_prox")], [], ("front", "top")),
}


# Per-samyuta joint overrides (degrees) applied to both hands. Anjali: the thumbs rest
# straight up side by side, and a slight 7 deg knuckle bend lets fingertips and palms
# meet together (the mesh palm is ~13 mm proud of the finger surfaces).
SAMYUTA_OVERRIDE = {"anjali": dict(thumb_cmc_flex=-10, thumb_cmc_abd=-45, thumb_cmc_rot=0, thumb_mcp=0, thumb_ip=0,
                                   index_mcp=7, middle_mcp=7, ring_mcp=7, little_mcp=7),
                    # Kapota: a shallower hood than single-hand Sarpashirsha so the fingertips meet
                    # without the fingers crossing; thumbs straight up as in Anjali
                    "kapota": dict(thumb_cmc_flex=-10, thumb_cmc_abd=-45, thumb_cmc_rot=0, thumb_mcp=0, thumb_ip=0,
                                   index_mcp=15, index_pip=20, index_dip=10, middle_mcp=15, middle_pip=20, middle_dip=10,
                                   ring_mcp=15, ring_pip=20, ring_dip=10, little_mcp=15, little_pip=20, little_dip=10)}


# ---------------------------------------------------------------- descriptions
# Formation rules paraphrased from the Abhinaya Darpana (the "book" the agent may
# consult with read_description, at a score penalty).
DESCRIPTIONS = {
    "pataka": "Pataka (flag): all four fingers extended and held close together; the thumb is bent and pressed against the side of the index finger.",
    "tripataka": "Tripataka (three parts of a flag): as Pataka, but the ring finger is bent down.",
    "ardhapataka": "Ardhapataka (half flag): as Tripataka, with the little finger also bent; index and middle fingers stay up, thumb alongside.",
    "kartarimukha": "Kartarimukha (scissors face): index and middle fingers extended and separated like scissor blades; ring and little fingers folded into the palm, held down by the thumb.",
    "mayura": "Mayura (peacock): the tip of the ring finger touches the tip of the thumb; the other fingers are extended and held close.",
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
    "mrigashirsha": "Mrigashirsha (deer head): index, middle and ring fingers held together and bent forward at the knuckles; the thumb and the little finger are raised straight up like horns.",
    "simhamukha": "Simhamukha (lion face): the tips of the middle and ring fingers touch the thumb; index and little fingers are extended.",
    "kangula": "Kangula (bell): the tips of the thumb, index and middle fingers are brought together; the ring finger is bent into the palm and the little finger stands erect.",
    "alapadma": "Alapadma (full-blown lotus): all fingers spread wide apart like a fan, each successive finger curving a little more toward the little finger side; thumb stretched out.",
    "chatura": "Chatura (clever): index, middle and ring fingers held together, the little finger extended apart; the thumb is bent across the palm to the base of the ring finger.",
    "bhramara": "Bhramara (bee): the middle finger tip touches the thumb tip, the index finger is curled, ring and little fingers are extended.",
    "hamsasya": "Hamsasya (swan's beak): the tips of the thumb and index finger touch; the other three fingers are extended and separated.",
    "hamsapaksha": "Hamsapaksha (swan's wing): index, middle and ring fingers held together, the thumb bent alongside; the little finger is lifted well away from the others.",
    "sandamsha": "Sandamsha (tongs): the curved fingers of Padmakosha are repeatedly opened and closed; in the closed moment the fingertips meet the thumb.",
    "mukula": "Mukula (bud): all five fingertips brought together at one point, like a closed bud.",
    "tamrachuda": "Tamrachuda (rooster): a fist with the index finger raised and hooked, the thumb pressing on the middle finger.",
    "trishula": "Trishula (trident): the thumb and little finger touch (little finger bent); index, middle and ring fingers stand up separated like a trident.",
    # samyuta
    "anjali": "Anjali (salutation): both hands in Pataka, palms pressed together, fingers pointing up.",
    "kapota": "Kapota (dove): as Anjali, but the palms are hollowed (Sarpashirsha); the fingertips and the bases of the palms meet, the centres stay apart.",
    "swastika": "Swastika (auspicious cross): two Pataka hands crossed at the wrists, held in profile, the fingers diverging upward in a narrow V.",
    "kartariswastika": "Kartariswastika (crossed scissors): two Kartarimukha hands crossed at the wrists.",
    "pushpaputa": "Pushpaputa (flower casket): two Sarpashirsha hands held palms up side by side, little-finger edges touching, as if offering flowers.",
}
