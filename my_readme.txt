Training Files: 
<deIdi2b2>: indicator for de-ID challenge. important for evaluation script
<TEXT>: patient note. wrapped in CDATA to be interpreted as one string
<TAGS>: PHI locations in <TEXT>. fields are specified in dtd file

Testing Files: 
Unlabeled- "PHI-noTags"
-> <TEXT>
Labeled- "PHI-Gold-fixed"
-> <TEXT> & <TAGS> (same as training)