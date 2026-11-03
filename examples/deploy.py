"""Deploy a logged MLflow model from a managed workspace or Argo container.

Install the updated mlflow-triton-control package in this environment first.
Triton Control supplies TRITON_CONTROL_URL, TRITON_CONTROL_TOKEN and MLflow
tracking credentials. No S3 URI, profile ID or requirements argument is needed.
"""

import argparse
import json

from mlflow.deployments import get_deploy_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="New deployment name")
    parser.add_argument("--model-uri", required=True, help="MLflow models:/ or runs:/ reference")
    parser.add_argument("--s3-profile-name", required=True, help="Your saved S3 profile name")
    parser.add_argument("--image", required=True, help="Triton serving image")
    parser.add_argument("--wait-timeout-seconds", type=float, default=300)
    args = parser.parse_args()

    client = get_deploy_client("triton-control")
    deployment = client.create_deployment(
        name=args.name,
        model_uri=args.model_uri,
        config={
            "s3_profile_name": args.s3_profile_name,
            "image": args.image,
            "wait_timeout_seconds": args.wait_timeout_seconds,
        },
    )
    print(json.dumps(deployment, indent=2))


if __name__ == "__main__":
    main()
