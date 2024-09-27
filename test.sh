#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# ### tar.gz to tar
# gunzip -c mbh_seg_container.tar.gz > mbh_seg_container.tar

# docker load -i mbh_seg_container.tar

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="32g"

docker volume create mbh_seg_container-output-$VOLUME_SUFFIX

echo $VOLUME_SUFFIX
echo $SCRIPTPATH
# Do not change any of the parameters to docker run, these are fixed
docker run -it --name mbh_seg_container-output-$VOLUME_SUFFIX  --gpus all --memory="${MEM_LIMIT}" --memory-swap="${MEM_LIMIT}" --network="none" --cap-drop="ALL" --security-opt="no-new-privileges" --shm-size="128m" --pids-limit="256" -v $SCRIPTPATH/input/:/opt/app/input -v $SCRIPTPATH/output_results:/opt/app/output  mbh_seg_container:latest 
          

# docker run --rm \
#         -v mbh_segcontainer-output-$VOLUME_SUFFIX:/output/ \
#         python:3.10-slim ls -al /model

# docker run --rm \
#         -v mbh_segcontainer-output-$VOLUME_SUFFIX:/output/ \
#         python:3.10-slim cat /output/results.json | python -m json.tool

# docker run --rm \
#         -v mbh_segcontainer-output-$VOLUME_SUFFIX:/output/ \
#         -v $SCRIPTPATH/test/:/input/ \
#         python:3.10-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

# if [ $? -eq 0 ]; then
#     echo "Tests successfully passed..."
# else
#     echo "Expected output was not found..."
# fi

docker volume rm mbh_seg_container-output-$VOLUME_SUFFIX
