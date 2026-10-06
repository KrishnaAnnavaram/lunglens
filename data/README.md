# Data

Git does not track the files in this folder, except this README. Do not commit CT scans,
patches, manifests or model weights.

## 1. Synthetic cohort (default, no download)

```bash
lunglens synth --out data/synthetic --patients 120 --seed 13
```

| File | Contents |
|---|---|
| `data/synthetic/manifest.jsonl` | One line per nodule: `patient_id`, `nodule_id`, `ratings`, `diameter_mm`, `patch_file` |
| `data/synthetic/patches/*.npy` | One 32 x 32 x 32 float32 patch per nodule, in Hounsfield units, 1 mm voxels |

The synthetic patches are not medical images. They are spheres with noise, made to test the code.

## 2. LIDC-IDRI (real data, optional)

| Item | Value |
|---|---|
| Name | LIDC-IDRI (Lung Image Database Consortium image collection) |
| Host | The Cancer Imaging Archive (TCIA) |
| URL | https://www.cancerimagingarchive.net/collection/lidc-idri/ |
| Licence | CC BY 3.0. Cite the collection and the data paper as TCIA asks |
| Size | 1010 patients, about 125 GB of DICOM |

Procedure:

1. Download the DICOM files with the NBIA Data Retriever.
2. Install the extra: `pip install -e ".[lidc]"`.
3. Write `~/.pylidcrc` with the DICOM folder (see the pylidc documentation).
4. Run `lunglens extract-lidc --out data/lidc --min-readers 1`.

The command writes `data/lidc/manifest.jsonl` and `data/lidc/patches/*.npy`. One record is one
pylidc annotation cluster (one physical nodule). `ratings` holds the malignancy rating (1-5) of
each reader who marked the nodule. A slice with no nodule never gets a label.

## 3. Manifest rules

| Field | Rule |
|---|---|
| `patient_id` | `LIDC-IDRI-NNNN` or `SYN-NNNN` |
| `nodule_id` | Unique in the file |
| `ratings` | 1 to 4 integers, each 1-5. There is no 0 |
| `diameter_mm` | Greater than 0 and less than 100 |
| `patch_file` | Path of the `.npy` patch, relative to the manifest |

Label rule: median rating 1-2 = benign (0), 4-5 = malignant (1), 3 (and 2.5 or 3.5) = indeterminate, not used.
