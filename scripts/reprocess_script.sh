#!/bin/bash

set -o xtrace

storms=(
    2018157N14255
    2019236N10314
    2020318N16289
    2022293N15259
    2023294N09264
    2018161N12260
    2019255N15251
    2021195N13249
    2023225N14248
    2024181N09320
    2018240N15251
    2019266N11341
    2021222N12262
    2023228N12259
    2024268N17278
    2018242N13343
    2020230N11264
    2021239N17281
    2023232N13300
    2024274N14328
    2018268N14253
    2020233N14313
    2021266N10327
    2023239N21274
    2024279N21265
    2018273N12259
    2020256N11329
    2022191N14249
    2023248N12257
    2024296N13260
    2018280N18273
    2020272N12257
    2022257N16312
    2023249N12320
    2018292N14261
    2020279N16284
    2022266N12294
    2023276N11254
    2019181N10252
    2020306N15288
    2022272N14258
    2023290N12256
    )

input_dir=pre-training/cyclones/goes
output_dir=pre-training-reprocessed/cyclones/goes
sat=goes

for storm in "${storms[@]}"; do
    input_dir_storm=$input_dir/$storm
    output_dir_storm=$output_dir/$storm
    python reprocess_pretraining.py --input $input_dir_storm --output $output_dir_storm --upload --sat $sat
done

