"""Control-only plate normalization; no phenotype/label-based sample filtering."""
import json
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

from intake import OUT, RAW, ASSAYS, ANNOTATIONS, canonical_metadata, digest, save_json


def main():
    t0 = time.time()
    schema = next(iter(json.loads((OUT / 'feature_schemas.json').read_text()).values()))
    # Remove segmentation identifiers and image position, but retain meaningful
    # neighborhood, skeleton, area and Euler-number morphology measurements.
    excluded = [c for c in schema if c.startswith('Metadata_') or
                any(s in c for s in ['BoundingBoxMaximum_', 'BoundingBoxMinimum_',
                                     '_Center_X', '_Center_Y', 'ClosestObjectNumber'])]
    features = [c for c in schema if c not in excluded]
    units = pd.read_csv(OUT / 'unit_partition_labels.csv')
    keys = units.loc[units[ASSAYS].notna().any(axis=1), 'parent_key'].sort_values().to_numpy()
    compounds = pd.read_csv(OUT / 'compound_identity.csv')
    wells = pd.read_csv(OUT / 'well_coverage.csv.gz').merge(
        compounds[['Metadata_EOS', 'parent_key']], on='Metadata_EOS', how='left', validate='many_to_one')
    mf = pd.read_csv(OUT / 'profile_file_manifest.csv')
    qc = []
    policy = dict(
        mode='source_control_normalization_no_label_fitting',
        count='log2((Metadata_Object_Count+1)/(plate DMSO median count+1)); actual segmented-object count, not proxy',
        morphology='(x-DMSO median)/(1.4826*MAD); zero MAD -> IQR/1.349; zero scale -> missing; clip [-20,20]',
        aggregation='median over observed technical wells within parent compound and context; preserve missingness',
        target_controls='DMSO wells from deployment plate permitted; no target treatment distribution or label fitting',
        excluded_columns=excluded, morphology_columns=features,
        sample_rule='576 exact-parent Tox21 matched compounds; no low-count or activity filtering',
        missing_rule='keep incomplete replicates; impute only inside downstream training folds',
        code_sha256=digest(__file_path())['sha256'])
    save_json('preprocessing_decisions.json', policy)
    with zipfile.ZipFile(RAW) as z:
        for context in ANNOTATIONS:
            rows, arrays, counts, rawcounts = [], [], [], []
            for n in mf.loc[(mf.context == context) & (mf.status == 'included'), 'member']:
                with z.open(n) as f:
                    d = canonical_metadata(pd.read_csv(f))
                m = wells[wells.source_member == n]
                idx = m.set_index('Metadata_Well').reindex(d.Metadata_Well)
                assert idx.source_member.notna().all()
                control = idx.Metadata_EOS.eq('DMSO').to_numpy()
                assert control.sum() >= 15
                x = d[features].to_numpy(float)
                x[~np.isfinite(x)] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    med = np.nanmedian(x[control], axis=0)
                    scale = 1.4826 * np.nanmedian(np.abs(x[control] - med), axis=0)
                    iqr = (np.nanpercentile(x[control], 75, axis=0) - np.nanpercentile(x[control], 25, axis=0)) / 1.349
                scale = np.where(scale > 1e-12, scale, iqr)
                scale[~np.isfinite(scale) | (scale <= 1e-12)] = np.nan
                x = np.clip((x - med) / scale, -20, 20).astype('float32')
                c = d.Metadata_Object_Count.to_numpy(float)
                assert np.isfinite(c).all() and (c >= 0).all()
                cm = np.median(c[control])
                keep = idx.parent_key.isin(keys).to_numpy()
                arrays.append(x[keep])
                counts.append(np.log2((c[keep]+1)/(cm+1)).astype('float32'))
                rawcounts.append(c[keep])
                rows.append(idx.loc[keep, ['parent_key', 'Metadata_Batch', 'Metadata_Plate', 'Metadata_EOS']].reset_index(drop=True))
                qc.append(dict(context=context, member=n, rows=len(d), dmso=int(control.sum()),
                               dmso_count_median=float(cm), undefined_scale_features=int(np.isnan(scale).sum()),
                               matched_wells=int(keep.sum()), zero_count_wells=int((c == 0).sum())))
            meta = pd.concat(rows, ignore_index=True)
            x, c, rawc = np.concatenate(arrays), np.concatenate(counts), np.concatenate(rawcounts)
            a, ac, ar, nr = [], [], [], []
            for key in keys:
                ix = meta.parent_key.eq(key).to_numpy()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    a.append(np.nanmedian(x[ix], axis=0) if ix.any() else np.full(len(features), np.nan))
                ac.append(float(np.median(c[ix])) if ix.any() else np.nan)
                ar.append(float(np.median(rawc[ix])) if ix.any() else np.nan)
                nr.append(int(ix.sum()))
            np.savez_compressed(OUT / (context + '_profiles.npz'),
                                parent_key=keys.astype(str), M=np.asarray(a, dtype='float32'),
                                C=np.asarray(ac, dtype='float32')[:, None], raw_count=np.asarray(ar),
                                observed_wells=np.asarray(nr), features=np.asarray(features))
            np.savez_compressed(OUT / (context + '_replicates.npz'),
                                parent_key=meta.parent_key.to_numpy(str),
                                replicate=meta.Metadata_Batch.to_numpy(str),
                                plate=meta.Metadata_Plate.to_numpy(str), M=x, C=c, raw_count=rawc)
            print(context, 'compounds', len(keys), 'observed', sum(n > 0 for n in nr),
                  'wells', len(meta), 'morphology_features', len(features), flush=True)
    pd.DataFrame(qc).to_csv(OUT / 'plate_normalization_qc.csv', index=False)
    save_json('profiles_preparation_summary.json', dict(
        status='complete_no_models_or_test_scores', seconds=time.time()-t0,
        features=len(features), compounds=len(keys), inputs=[digest(OUT / 'unit_partition_labels.csv'), digest(RAW)],
        outputs=[digest(p) for p in sorted(OUT.glob('*_profiles.npz'))]))


def __file_path():
    from pathlib import Path
    return Path(__file__)


if __name__ == '__main__':
    main()
