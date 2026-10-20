# test_small: 12-sample subset of the test set

Two samples per test subject (pp1, pp27, pp30, pp53, pp65, pp67), taken from the full 599-sample set, for a quick check
that the code runs and gives the expected numbers. Same layout as the full test set:

| Folder | Content |
|---|---|
| `brir/` | simulated BRIRs (SofaMyRoom, from the HUTUBS HRIRs) |
| `hrir/`, `doa/`, `anthropometry/` | ground-truth HRIR, direction of arrival, and anthropometry of each sample |
| `speech_48k/` | VCTK utterance (speaker p226 or p287) used as the dry source; listed in `speech_files.csv` |
| `baselines/` | Random / Generic / Nearest Neighbour estimates (so no HUTUBS SOFA files are needed to evaluate) |

## Sources
- HUTUBS HRTF database, TU Berlin: F. Brinkmann et al., "The HUTUBS head-related transfer function (HRTF) database",
  2019, https://depositonce.tu-berlin.de/handle/11303/9429
- CSTR VCTK Corpus 0.92, University of Edinburgh (CC BY 4.0): J. Yamagishi, C. Veaux, K. MacDonald,
  https://datashare.ed.ac.uk/handle/10283/3443
