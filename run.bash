#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
API_PORT=8000
UI_PORT=7860
CLUSTERS=10
max_tag_distance=0.5


# python3 main.py \
#         --api_port $API_PORT \
#         --ui_port $UI_PORT \
#         --n_clusters $CLUSTERS \
#         --max_tag_distance $max_tag_distance \
#         --is_kmeans \
#         --enable_ui  

python3 main.py \
        --api_port $API_PORT \
        --ui_port $UI_PORT \
        --n_clusters $CLUSTERS \
        --max_tag_distance $max_tag_distance \
        --enable_ui  
