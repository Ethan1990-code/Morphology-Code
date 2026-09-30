"""EUOS v3: source-qualified wells and pre-performance historical holdout.

Raw archives stay read-only. Never executes code from source archives.
"""
import hashlib
import io
import json
import platform
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

OUT = Path(__file__).resolve().parent
P = OUT.parents[1]
FEAS = OUT.parent / 'EUOS_feasibility_2026-09-21_v1'
JUMP = OUT.parent / 'JUMP_Tox21_feasibility_2026-09-07_v1'
RAW = P / 'data/Aggregated_Profiles.zip'
OLD = FEAS / 'Profile_Analysis_v1.zip'
ASSAYS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
          'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
ANNOTATIONS = {
    'FMP_HepG2': '2022-07-08_Annotation_Bioactives_HepG2.csv',
    'FMP_U2OS': '2023-05-23_Annotation_Bioactives_U2OS_Corrected.csv',
    'IMTM_HepG2': '2023-08-14_Annotation2_IMTM_HepG2.csv',
    'MEDINA_HepG2': '2023-11-28_Annotation_MEDINA_HepG2.csv',
    'USC_HepG2': '2023-11-28_Annotation_USC_HepG2.csv',
}
STALE = '2023-07-07_HepG2_10uM_B1001_R4_CP_Profiles_Aggregated.csv'
KEYS = ['Metadata_Plate', 'Metadata_Batch', 'Metadata_Well']
rdBase.DisableLog('rdApp.*')


def digest(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            sha.update(b)
            md5.update(b)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest())


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def identity(mol):
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        Chem.SanitizeMol(mol)
        key = Chem.MolToInchiKey(mol)
        # Achiral scaffold groups intentionally keep stereoisomers together.
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return dict(parent_key=key, connectivity=key.split('-')[0],
                    scaffold=scaffold or 'ACYCLIC::' + key.split('-')[0],
                    parent_smiles=Chem.MolToSmiles(mol))
    except Exception:
        return None


def canonical_metadata(df):
    df = df.copy()
    combined = df.Metadata_Plate.str.extract(r'^(B\d+)_(R\d+)$')
    mask = combined[0].notna()
    df.loc[mask, 'Metadata_Plate'] = combined.loc[mask, 0]
    df.loc[mask, 'Metadata_Batch'] = combined.loc[mask, 1]
    assert df.Metadata_Plate.str.fullmatch(r'B100[1-7]').all()
    assert df.Metadata_Batch.str.fullmatch(r'R[1-4]').all()
    assert not df.duplicated(KEYS).any()
    return df


def metadata():
    t0 = time.time()
    rawhash = digest(RAW)
    assert rawhash['md5'] == '69885f22ea74fd61e062ab27191b4f8e'
    with zipfile.ZipFile(OLD) as z:
        anns = {k: pd.read_csv(io.BytesIO(z.read('annotations/' + n))) for k, n in ANNOTATIONS.items()}
        compounds = pd.read_csv(io.BytesIO(z.read('annotations/pd_export_04_2022_2464_compounds_standardized.csv')))
        eos = pd.read_csv(io.BytesIO(z.read('annotations/2024-08-02_EOS_pdid.csv')))
    ids = pd.DataFrame([identity(Chem.MolFromSmiles(s)) for s in compounds.smiles])
    assert len(ids) == 2464 and ids.parent_key.notna().all()
    compounds = pd.concat([compounds[['pdid', 'name']], ids], axis=1).merge(
        eos, left_on='pdid', right_on='Metadata_pdid', validate='one_to_one')
    compounds.to_csv(OUT / 'compound_identity.csv', index=False)
    manifests, wells, schemas = [], [], {}
    with zipfile.ZipFile(RAW) as z:
        names = [n for n in z.namelist() if n.endswith('.csv')]
        for i, n in enumerate(names):
            context = n.split('/')[-2]
            if context == 'IMTM_HepG2' and Path(n).name == STALE:
                manifests.append(dict(member=n, context=context, status='excluded_stale_pre_correction_duplicate'))
                continue
            with z.open(n) as f:
                header = pd.read_csv(f, nrows=0).columns.tolist()
            schemas[n] = header
            with z.open(n) as f:
                d = pd.read_csv(f, usecols=lambda c: c.startswith('Metadata_'))
            d = canonical_metadata(d)
            a = anns[context]
            joined = d.merge(a, on=KEYS, how='left', validate='one_to_one', indicator=True)
            assert joined._merge.eq('both').all(), n
            joined = joined.drop(columns='_merge')
            joined['context'], joined['source_member'] = context, n
            wells.append(joined)
            manifests.append(dict(member=n, context=context, status='included', rows=len(d),
                                  columns=len(header), crc32=z.getinfo(n).CRC,
                                  plate=d.Metadata_Plate.iloc[0], replicate=d.Metadata_Batch.iloc[0],
                                  dmso_wells=int(joined.Metadata_EOS.eq('DMSO').sum()),
                                  invalid_counts=int((~np.isfinite(d.Metadata_Object_Count) | (d.Metadata_Object_Count < 0)).sum())))
            if (i + 1) % 28 == 0:
                print('metadata files', i + 1, flush=True)
        # Full numeric equality verifies the discarded file duplicates corrected B1007, not an extra replicate.
        old_name = next(n for n in names if '/IMTM_' in n and Path(n).name == STALE)
        corrected = next(n for n in names if '/IMTM_' in n and '2023-07-07_HepG2_10uM_B1007_R4' in n)
        with z.open(old_name) as f:
            x = pd.read_csv(f)
        with z.open(corrected) as f:
            y = pd.read_csv(f)
        numeric = [c for c in x.columns if not c.startswith('Metadata_') or c == 'Metadata_Object_Count']
        duplicate_max_difference = float(np.nanmax(np.abs(x[numeric].to_numpy() - y[numeric].to_numpy())))
        duplicate_equal = (x.Metadata_Well.equals(y.Metadata_Well)
                           and x.Metadata_Object_Count.equals(y.Metadata_Object_Count)
                           and np.allclose(x[numeric], y[numeric], rtol=1e-10, atol=1e-10, equal_nan=True))
        assert duplicate_equal
        (OUT / 'source_correction_readme.txt').write_bytes(z.read(next(n for n in z.namelist() if n.endswith('README.txt'))))
    wd = pd.concat(wells, ignore_index=True)
    assert not wd.duplicated(['context'] + KEYS).any()
    wd.to_csv(OUT / 'well_coverage.csv.gz', index=False)
    pd.DataFrame(manifests).to_csv(OUT / 'profile_file_manifest.csv', index=False)
    save_json('feature_schemas.json', schemas)
    # History means actual analyzed units, not every downloaded compound.
    hist, history_audit, history_paths = [], [], []
    jump = pd.read_csv(JUMP / 'analysis_ready_units.csv')
    jump_map = pd.read_csv(JUMP / 'inputs/jump_compound.csv.gz')
    jm = jump_map[jump_map.Metadata_InChIKey.str.split('-').str[0].isin(set(jump.connectivity_key))]
    for source, seq in [
        ('JUMP', [Chem.MolFromSmiles(s) for s in jm.Metadata_SMILES]),
        ('EveBio', [Chem.MolFromInchi(s) for s in pd.read_csv(P / 'data/positive_controls_2026-09-08/EveBio/CP_all_EveBio.csv').InChICode_standardised]),
        ('Cardiac', [Chem.MolFromSmiles(s) for s in pd.read_csv(P / 'analysis/A1_negative_transfer_and_candidate_scan_2026-09-06_v2/full_structure_oof_predictions.csv.gz').parent_smiles])]:
        results = [identity(m) for m in seq]
        history_audit.append(dict(source=source, rows=len(seq), invalid=sum(r is None for r in results)))
        hist.extend(dict(source=source, **r) for r in results if r)
    hd = pd.DataFrame(hist).drop_duplicates(['source', 'parent_key'])
    hd.to_csv(OUT / 'historical_identity_union.csv', index=False)
    history_paths = [JUMP / 'analysis_ready_units.csv', JUMP / 'inputs/jump_compound.csv.gz',
                     P / 'data/positive_controls_2026-09-08/EveBio/CP_all_EveBio.csv',
                     P / 'analysis/A1_negative_transfer_and_candidate_scan_2026-09-06_v2/full_structure_oof_predictions.csv.gz']
    historic_conn = set(hd.connectivity) | set(jump.connectivity_key)
    historic_scaf = set(hd.scaffold) | set(jump.murcko_scaffold.dropna())
    toxpath = JUMP / 'inputs/tox21_10k_data_all.sdf.zip'
    tox, bad = [], 0
    with zipfile.ZipFile(toxpath) as z:
        n = next(n for n in z.namelist() if n.endswith('.sdf'))
        with z.open(n) as f:
            for m in Chem.ForwardSDMolSupplier(f, removeHs=False):
                row = identity(m)
                if row is None:
                    bad += 1
                    continue
                if row['parent_key'] not in set(compounds.parent_key):
                    continue
                row.update({a: m.GetProp(a).strip() if m.HasProp(a) else '' for a in ASSAYS})
                tox.append(row)
    tx = pd.DataFrame(tox)
    units = compounds.drop_duplicates('parent_key')[['parent_key', 'connectivity', 'scaffold', 'parent_smiles', 'name']].copy()
    conflicts = []
    for a in ASSAYS:
        vals = tx[tx[a].isin(['0', '1'])].groupby('parent_key')[a].agg(lambda s: sorted(set(s)))
        units[a] = units.parent_key.map({k: int(v[0]) for k, v in vals.items() if len(v) == 1})
        conflicts.append(dict(assay=a, conflicting_keys=int(vals.map(len).gt(1).sum())))
    units['historical_entity'] = units.connectivity.isin(historic_conn)
    units['historical_scaffold'] = units.scaffold.isin(historic_scaf)
    units['partition'] = np.select([units.historical_entity, units.historical_scaffold],
                                   ['development', 'new_entity_seen_scaffold'], default='strict_holdout')
    coverage = wd.merge(compounds[['Metadata_EOS', 'parent_key']], on='Metadata_EOS', how='inner', validate='many_to_one')
    cov = coverage.groupby(['parent_key', 'context']).Metadata_Batch.nunique().unstack(fill_value=0)
    for context in ANNOTATIONS:
        units['replicates_' + context] = units.parent_key.map(cov[context]).fillna(0).astype(int)
    units['all_contexts_observed'] = units[['replicates_' + c for c in ANNOTATIONS]].ge(1).all(axis=1)
    units['all_hepg2_observed'] = units[['replicates_' + c for c in ANNOTATIONS if c.endswith('HepG2')]].ge(1).all(axis=1)
    units.to_csv(OUT / 'unit_partition_labels.csv', index=False)
    qual = []
    for a in ASSAYS:
        info = dict(assay=a)
        for part in ['development', 'strict_holdout', 'new_entity_seen_scaffold']:
            sub = units[units.partition.eq(part) & units.all_hepg2_observed & units[a].notna()]
            info.update({part + '_n': len(sub), part + '_positive': int(sub[a].sum()),
                         part + '_negative': int((sub[a] == 0).sum()), part + '_groups': sub.scaffold.nunique()})
        info['eligible_primary'] = all(info[p + '_positive'] >= 20 and info[p + '_negative'] >= 20 and info[p + '_groups'] >= 10
                                       for p in ['development', 'strict_holdout'])
        qual.append(info)
    qt = pd.DataFrame(qual)
    qt.to_csv(OUT / 'endpoint_qualification.csv', index=False)
    cohort_counts = units[units[ASSAYS].notna().any(axis=1)].partition.value_counts().to_dict()
    summary = dict(status='qualified_metadata_no_model_performance_read', date='2026-09-21',
                   inputs=[rawhash, digest(OLD), digest(toxpath)] + [digest(f) for f in history_paths],
                   source='https://zenodo.org/records/19347244', license='CC-BY-4.0',
                   annotation_version='v1; exact plate/replicate/well join to v3, IMTM source correction respected',
                   included_profile_files=140, excluded_stale_files=1, stale_numeric_copy_confirmed=duplicate_equal,
                   stale_copy_max_absolute_rounding_difference=duplicate_max_difference,
                   total_observed_wells=len(wd), wells_by_context=wd.context.value_counts().to_dict(),
                   missing_wells_by_context={c: int(10752 - wd.context.eq(c).sum()) for c in ANNOTATIONS},
                   unique_parent=units.parent_key.nunique(), labeled_any=int(units[ASSAYS].notna().any(axis=1).sum()),
                   history_audit=history_audit, labeled_partition_counts=cohort_counts,
                   primary_eligible=qt.loc[qt.eligible_primary, 'assay'].tolist(),
                   conflicting_labels=conflicts, tox_bad_structures=bad,
                   oasis_policy='Prior PoD descriptive case only, not fitted or used to tune candidate fusion selection. Not asserted as independent validation. No OASIS model-training units exist in this protocol.',
                   feature_policy='No outcome or morphology activity filter. Shape excludes direct count metadata; train-only feature processing later.',
                   software=dict(python=platform.python_version(), pandas=pd.__version__, rdkit=rdBase.rdkitVersion),
                   runtime_seconds=time.time()-t0)
    save_json('intake_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['total_observed_wells', 'missing_wells_by_context', 'labeled_partition_counts', 'primary_eligible', 'history_audit']}, ensure_ascii=False, indent=2))
    print(qt.to_string(index=False), flush=True)


if __name__ == '__main__':
    metadata()
