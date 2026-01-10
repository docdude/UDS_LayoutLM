#!/bin/bash
# filepath: start_label_studio.sh

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\\Users\\jloya\\Documents\\UDS_LayoutLM\\data"

echo "Starting Label Studio with local files enabled..."
echo "Document root: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"

label-studio start