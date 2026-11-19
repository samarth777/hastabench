# HastaBench v1 target verification (2026-11-19)

Every target was compared against >= 4 independent reference images (dancer photos, dance-school charts,
Wikimedia Commons, natyasutraonline gallery, onlinebharatanatyam, and the 1917 *Mirror of Gesture* text/plates).
Per-group details with source URLs: g1_summary.md ... g5_summary.md.

| verdict | single-hand (28) | two-hand (5) |
|---|---|---|
| CORRECT | 8: kartarimukha, ardhachandra, katakamukha, chandrakala, simhamukha, bhramara, hamsasya, trishula | 1: pushpaputa |
| MINOR | 17: pataka, tripataka, ardhapataka, mayura, arala, shukatunda, mushti, shikhara, kapittha, suchi, padmakosha, sarpashirsha, alapadma, chatura, hamsapaksha, mukula, tamrachuda | 3: anjali, kapota, kartariswastika |
| WRONG | 3: mrigashirsha, kangula, sandamsha | 1: swastika |

8 + 17 + 3 = 28.

## Systematic issues (fix once, many hastas improve)
1. `T_OVER` thumb (mushti, suchi, tamrachuda, kartarimukha) too straight -> thumb protrudes from the fist. Use cmc_flex 40, cmc_abd -15, cmc_rot 25, mcp 50, ip 35.
2. `TOGETHER` spread too loose -> gaps between fingers in Pataka family. Use (-7, 0, 9, 13) or contact-tight.
3. `B` (bent) too deep for Tripataka/Ardhapataka -> use mcp 35, pip 95, dip 50 (tip points forward, not into palm).
4. `S` (Sarpashirsha hood) too shallow -> mcp 25, pip 30, dip 20.
5. Flexion in references is concentrated at the MCP more than our intents assume.
6. Display: Shikhara's roll-90 view hides the thumb in the front camera.

## Wrong targets
- mrigashirsha: index/middle/ring must fold ~80 deg at MCP; little finger upright (not splayed); thumb erect.
- kangula: practice = thumb+index+middle tips pinched, ring folded, little erect (not a padmakosha claw).
- sandamsha: Abhinaya Darpana form = Padmakosha opening/closing (dynamic); static proxy = closed frame with all tips meeting the thumb.
- swastika: Pataka hands crossed at the wrists held in profile (palms toward the midline), narrow V; ours faces the audience.

## Model limitations surfaced
- little_abd limit +-25 deg too small for Hamsapaksha / Alapadma fans.
- Forearm is coaxial with the hand, so horizontal-forearm + bent-wrist poses (Swastika) cannot be shown exactly.
- Several hastas have genuine regional variants (Arala thumb, Hamsasya spread, Chatura straight vs bent, Mrigashirsha MCP-only vs curled).
