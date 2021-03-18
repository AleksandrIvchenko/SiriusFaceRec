#!/bin/sh

name="$1"
gsed -i "s/protostar/$name/g" \
    setup.py \
    protostar/datamodules/protostar_datamodule.py \
    protostar/loggers/neptune_logger.py \
    protostar/models/protostar_model.py \
    protostar/trainer/trainer.py \
    scripts/fit_protostar.py \
    scripts/install_module.sh
mv protostar $name

