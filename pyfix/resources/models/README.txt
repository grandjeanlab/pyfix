Pre-trained pyfix models

This directory contains a collection of pre-trained pyfix models. These models
can be used for automatic classification of ICs for datasets which have
similar properties (e.g. resolution, TR, spatial smoothing, etc).

The info.json file contains an index of all built in models, along with some
associated settings.

 - Standard.pyfix_model: for use on more "standard" FMRI datasets / analyses;
   e.g., TR=3s, Resolution=3.5x3.5x3.5mm, Session=6mins, default FEAT
   preprocessing (including default spatial smoothing).

 - HCP25_hp2000.pyfix_model: for use on "minimally-preprocessed" 3T HCP-like
   datasets, e.g., TR=0.7s, Resolution=2x2x2mm, Session=15mins, no spatial
   smoothing, minimal (2000s FWHM) highpass temporal filtering.

 - HCP7T_hp2000.pyfix_model: for use on "minimally-preprocessed" 7T HCP-like
   datasets, e.g., TR=1.0s, Resolution=1.6x1.6x1.6mm, Session=15mins, no
   spatial smoothing, minimal (2000s FWHM) highpass temporal filtering.

 - HCP_Style_Single_Multirun_Dedrift.pyfix_model: drived from task and resting
   state fMRI data from 75 HCP YA 3T data sets.

 - WhII_MB6.pyfix_model: derived from the Whitehall imaging study, using
   multiband x6 EPI acceleration: TR=1.3s, Resolution=2x2x2mm, Session=10mins,
   no spatial smoothing, 100s FWHM highpass temporal filtering.

 - WhII_Standard.pyfix_model: derived from more traditional early parallel
   scanning in the Whitehall imaging study, using no EPI acceleration: TR=3s,
   Resolution=3x3x3mm, Session=10mins, no spatial smoothing, 100s FWHM
   highpass temporal filtering.

 - UKBiobank.pyfix_model: derived from fairly HCP-like scanning in the UK
   Biobank imaging study: 40 subjects, TR=0.735s, Resolution=2.4x2.4x2.4mm,
   Session=6mins, no spatial smoothing, 100s FWHM highpass temporal filtering.

 - NHPHCP_Macaque_RIKEN30SRFIX.pyfix_model: derived from NHP-HCP Macaque data,
   using multiband x5 EPI acceleration: TR=0.76s, Resolution:1.25x1.25x1.25mm,
   Session=102min, no spatial smoothing, minimal (2000s FWHM) highpass
   temporal filtering.

 - NHPHCP_MacaqueCyno_BOLD.pyfix_model: derived from NHP-HCP Cynomolgus
   Macaque data.

The data and hand labels for these models have been contributed by a large
number of people, including:

 - Gholamreza Salimi Khorshidi
 - Ludovica Griffanti
 - Gwenaelle Douaud
 - David Flitney
 - Claire Sexton
 - Eniko Zsoldos
 - Klaus Ebmeier
 - Nicola Filippini
 - Clare Mackay
 - Stephen Smith (Oxford)
 - Matthew Glasser
 - Donna Dierker
 - Erin Reid
 - David Van Essen (WashU)
 - Edward Auerbach
 - Steen Moeller
 - Junqian Xu
 - Essa Yacoub,
 - Kamil Ugurbil (Minnesota)
 - Giuseppe Baselli (Milan)
 - Christian Beckmann (Donders)
 - Takuya Hayashi
