"""One declared post-scoring technical correction, never outcome-selected."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
INTAKE=HERE.parents[1]/'preprocessing/EUOS_profiles_intake_2026-09-21_v1'
ASSAYS=['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
u=pd.read_csv(INTAKE/'unit_partition_labels.csv')
before=u.copy()
# Same connectivity identity may have source-dependent tautomers and therefore
# different raw Murcko representations. Include BOTH observed representations.
scaffolds=set(u.loc[u.historical_entity,'scaffold'])
u['historical_scaffold']=u.historical_scaffold | u.scaffold.isin(scaffolds)
u['partition']=np.select([u.historical_entity,u.historical_scaffold],['development','new_entity_seen_scaffold'],default='strict_holdout')
changed=u[u.partition.ne(before.partition)]
assert len(changed)==1 and changed.name.iloc[0]=='CEFOXITIN'
assert not set(u.loc[u.partition.eq('strict_holdout'),'scaffold']) & set(u.loc[u.partition.eq('development'),'scaffold'])
u.to_csv(HERE/'unit_partition_labels.csv',index=False)
rows=[]
for assay in ASSAYS:
    row={'assay':assay}
    for part in ['development','strict_holdout','new_entity_seen_scaffold']:
        s=u[u.partition.eq(part)&u.all_hepg2_observed&u[assay].notna()]
        row.update({part+'_n':len(s),part+'_positive':int(s[assay].sum()),part+'_negative':int(s[assay].eq(0).sum()),part+'_groups':s.scaffold.nunique()})
    row['eligible_primary']=all(row[p+'_positive']>=20 and row[p+'_negative']>=20 and row[p+'_groups']>=10 for p in ['development','strict_holdout'])
    rows.append(row)
pd.DataFrame(rows).to_csv(HERE/'endpoint_qualification.csv',index=False)
receipt=dict(status='post_scoring_identity_representation_correction',
    original_partition_sha256=hashlib.sha256((INTAKE/'unit_partition_labels.csv').read_bytes()).hexdigest(),
    corrected_partition_sha256=hashlib.sha256((HERE/'unit_partition_labels.csv').read_bytes()).hexdigest(),
    affected_keys=changed[['parent_key','name','partition']].to_dict('records'),
    method='Add EUOS representations of all previously seen connectivity entities to historical scaffold exclusion union; no model-score-driven selection.',
    changed_hyperparameters=False,changed_rules=False,changed_final_site=False,
    prior_results_retained='../EUOS_locked_fusion_evaluation_2026-09-21_v1',
    blinding='Original outer results were seen before this correction; v2 is a transparent corrected reanalysis, not a fresh independent blind test.',
    profile_policy='Reuse unchanged v1 raw-derived NPZ profiles; corrected partition manifests only.')
(HERE/'identity_correction_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
print(pd.DataFrame(rows).query('eligible_primary').to_string(index=False))
