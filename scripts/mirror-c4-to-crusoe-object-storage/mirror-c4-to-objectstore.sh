#!/bin/bash

# add your Crusoe Cloud Object Storage credentials
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""

# update to match your Crusoe Cloud region
export S3_ENDPOINT_URL="https://object.eu-iceland1-a.crusoecloudcompute.com"
export S3_BUCKET="c4-bucket"
export HF_TOKEN=""

kubectl create configmap mirror-c4-script --from-file=script.py --dry-run=client -o yaml | kubectl apply -f -

envsubst < job.yaml | kubectl apply -f -

echo "Waiting for job pod to appear..."
until POD=$(kubectl get pods -l job-name=mirror-c4-to-objectstorage --no-headers \
      -o custom-columns=':metadata.name' 2>/dev/null | head -1) && [ -n "$POD" ]; do
  sleep 2
done

echo "Waiting for pod $POD to reach Running state..."
until [[ $(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null) =~ ^(Running|Succeeded|Failed)$ ]]; do
  sleep 2
done

echo "Tailing logs for pod $POD ..."
kubectl logs -f "$POD"
