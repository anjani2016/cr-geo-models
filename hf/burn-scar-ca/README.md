---
license: apache-2.0
library_name: terratorch
base_model: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars
pipeline_tag: image-segmentation
tags:
  - earth-observation
  - remote-sensing
  - sentinel-2
  - wildfire
  - burned-area
  - semantic-segmentation
  - canada
  - prithvi
---

# burn-scar-ca v1: burned area in Sentinel-2 imagery of Canadian wildfires

Marks each 30 m pixel of a Sentinel-2 L2A image taken after a fire as **burned** or **not burned**. The model was fine-tuned from
[Prithvi-EO-2.0-300M-BurnScars](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars), which was trained on US fires,
on perimeters of Canadian fires.

**Output tier: `Modeled (Simulation)`.** A satellite-derived model estimate of burned area, not a measurement or a field survey.

Made by Centauri Research (Alpha Centauri Ventures Inc.). Code: [github.com/anjani2016/cr-geo-models](https://github.com/anjani2016/cr-geo-models).

## Why it exists
On Canadian fires held out from training, the US model found nearly every burned pixel. It also marked about 15% of unburned land as burned
(not-burned class accuracy 0.849, so 1 − 0.849 ≈ 15% false burns). Fine-tuning on Canadian fires cut those false burns.

## Held-out test (Quebec boreal fires, 274 chips)
Every fire inside a Quebec boreal region box was held out. Those chips were never used for training or tuning, and were used once, for this test.
Scores come from TerraTorch `trainer.test` on the same chips for both models.

| Metric | US base model (baseline) | burn-scar-ca v1 |
|---|---|---|
| Burn-scar IoU | 0.566 | **0.873** |
| Not-burned IoU | 0.846 | 0.972 |
| mIoU | 0.706 | **0.922** |
| Burned pixels found (recall) | 0.980 | 0.939 |
| Not-burned pixels kept (class accuracy) | 0.849 | **0.984** |
| Boundary mIoU | 0.060 | 0.559 |
| F1 | 0.820 | 0.959 |

In plain words: false burns fell from about 15% to under 2% of unburned pixels. The model misses a few more burned pixels than the
US model (recall 0.980 → 0.939). Overlap with the reference perimeters rose from 0.566 to 0.873.

## Limits
- **The reference is NBAC perimeters, not field-checked burn extent.** NBAC counts unburned islands inside a perimeter as burned, and its edges
  can be off by a pixel or two. The scores measure agreement with NBAC.
- **One held-out region.** Quebec boreal fires are one test of transfer. Performance elsewhere in Canada, in other ecosystems, seasons or
  sensors has not been measured.
- **Epoch-10 weights.** The cloud GPU runtime ended twice before training finished. These are the best weights pulled off the machine before
  it ended (validation loss 0.0572), not the run's best epoch. A longer run may score differently.
- **Imagery timing.** Each chip uses the least cloudy Sentinel-2 scene (under 20% scene cloud cover) within 45 days after the fire's end date
  (from 1 September of the fire year when NBAC gives no end date). Other timings have not been measured.
- **Burned / not burned only.** No severity classes in this version.
- Not for operational fire management, insurance, legal or regulatory decisions.

## Training data
| | |
|---|---|
| Labels | NBAC (National Burned Area Composite), release 20260513, fires 2017–2025. 1 = inside the fire-year perimeter; 0 = outside; −1 (ignored) = cloud, shadow, snow or no data, plus scars from the 3 previous years |
| Imagery | Sentinel-2 Level-2A surface reflectance (Cloud-Optimized GeoTIFFs found through the Earth Search STAC API, hosted on AWS), scenes from 2017–2026, resampled to 30 m |
| Bands, in order | BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2 (B02, B03, B04, B8A, B11, B12), reflectance on a 0–1 scale |
| Chips | 1,024 chips of 512 × 512 pixels: train 607, validation 143, test 274 |
| Split | By fire and region, never by random chip, so chips of one fire never sit on both sides of a split |
| Normalisation | `norm_stats.json`: band means and standard deviations from 149,510,593 valid training-chip pixels |

## Training
- **Base model:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`, revision `a3f2c410e45b8ac7417976614528a872f024d831`.
- **Framework:** TerraTorch 1.2.14, PyTorch 2.14.1, Lightning 2.6.6, on one NVIDIA L4 GPU (Google Colab).
- **Settings:** all weights trainable; AdamW, learning rate 1e-4; ReduceLROnPlateau; cross-entropy loss with −1 ignored; bf16 mixed precision.
  Full settings: `config/`.
- **Tracking:** every run, including the two cut short, was logged to MLflow with the code commit, base-model revision, data-manifest
  checksum and seed.

## Files
| File | What |
|---|---|
| `burn_scar_ca_v1_fp16.safetensors` | Weights of the TerraTorch `SemanticSegmentationTask`, float16 (cast to float32 to run). SHA-256 `ffd8e9c692b20202674431df8805e2e104a4d632be582b193fea054a5120425a` |
| `norm_stats.json` | Band means and standard deviations to normalise inputs |
| `config/` | The base TerraTorch config and this project's training and data-staging configs |
| `inference_example.py` | Loads the model and writes a burned / not-burned mask for one chip |
| `LICENSE`, `NOTICE` | Apache-2.0 licence; attributions |
| `SHA256SUMS` | Checksums of the files above |

The safetensors file was converted from the registered PyTorch state_dict (SHA-256 `b471b653b076bac8ec92c80e25d31fe0659996d06176f1c8d9a982ee9c924381`).
All 355 tensors were compared after conversion and are identical.

## How to use
```bash
pip install terratorch==1.2.14 rasterio safetensors
python inference_example.py chip.tif burned_mask.tif
```
The input is a six-band GeoTIFF in the band order above, reflectance on a 0–1 scale, at 30 m. No-data (NaN) pixels are set to 0 before
normalisation, as in training.

## Licence and attribution
- **Weights:** Apache-2.0. See `LICENSE` and `NOTICE`.
- **Base model:** Prithvi-EO-2.0-300M-BurnScars, Apache-2.0.
- **Labels:** Canadian Forest Service. National Burned Area Composite (NBAC). Natural Resources Canada, Canadian Forest Service, Northern Forestry
  Centre, Edmonton, Alberta. https://cwfis.cfs.nrcan.gc.ca. Contains information licensed under the
  [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
- **Imagery:** Contains modified Copernicus Sentinel data 2017–2026.

Use of this model is not an endorsement by any data or model provider.

## Citation
This model:
```bibtex
@software{burn_scar_ca_v1,
  author    = {Duddukuru, Anjani},
  title     = {burn-scar-ca v1: burned-area segmentation for Canadian wildfires in Sentinel-2 imagery},
  publisher = {Alpha Centauri Ventures Inc. (Centauri Research)},
  year      = {2026},
  url       = {https://huggingface.co/centauri-research2019/burn-scar-ca}
}
```
A DOI for the code release is added here once it is archived on Zenodo.

Base model:
```bibtex
@article{Prithvi-EO-V2-preprint,
  author  = {Szwarcman, Daniela and Roy, Sujit and Fraccaro, Paolo and Gíslason, Þorsteinn Elí and Blumenstiel, Benedikt and Ghosal, Rinki and de Oliveira, Pedro Henrique and de Sousa Almeida, João Lucas and Sedona, Rocco and Kang, Yanghui and Chakraborty, Srija and Wang, Sizhe and Kumar, Ankur and Truong, Myscon and Godwin, Denys and Lee, Hyunho and Hsu, Chia-Yu and Akbari Asanjan, Ata and Mujeci, Besart and Keenan, Trevor and Arévolo, Paulo and Li, Wenwen and Alemohammad, Hamed and Olofsson, Pontus and Hain, Christopher and Kennedy, Robert and Zadrozny, Bianca and Cavallaro, Gabriele and Watson, Campbell and Maskey, Manil and Ramachandran, Rahul and Bernabe Moreno, Juan},
  title   = {{Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications}},
  journal = {arXiv preprint arXiv:2412.02732},
  year    = {2024}
}
```
