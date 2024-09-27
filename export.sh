#!/usr/bin/env bash
docker save mbh_seg_container | gzip -c > mbh_seg_container.tar.gz
