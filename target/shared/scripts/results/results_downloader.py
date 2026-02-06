##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

"""
S3 downloader for Cylon experiment result files.

Downloads summary CSVs from S3 using prefix-based discovery,
or scans local directories for already-downloaded results.
"""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def download_from_s3(
    bucket: str,
    prefix: str,
    download_dir: str,
) -> List[str]:
    """Download all summary files from S3 matching a prefix.

    Returns list of local file paths downloaded.
    """
    import boto3
    from botocore.exceptions import ClientError

    s3_client = boto3.client('s3')
    downloaded = []

    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                if not filename.startswith('cylon_summary_'):
                    continue
                if filename.endswith('.log'):
                    continue

                local_path = os.path.join(download_dir, filename)
                os.makedirs(download_dir, exist_ok=True)

                logger.info(f"Downloading s3://{bucket}/{key} -> {local_path}")
                s3_client.download_file(bucket, key, local_path)
                downloaded.append(local_path)

    except ClientError as e:
        logger.error(f"S3 error: {e}")
    except Exception as e:
        logger.error(f"Download error: {e}")

    logger.info(f"Downloaded {len(downloaded)} files from s3://{bucket}/{prefix}")
    return downloaded


def download_experiment_results(config) -> None:
    """Download all summary files for configured experiments from S3.

    Populates local_data_dir on each experiment config after download.
    """
    for exp in config.experiments:
        if exp.local_data_dir:
            logger.info(f"Skipping S3 download for {exp.label} (using local: {exp.local_data_dir})")
            continue

        if not exp.s3_prefix_pattern or not config.s3_bucket:
            logger.warning(f"No S3 config for {exp.label}, skipping download")
            continue

        prefix = exp.s3_prefix_pattern.format(
            platform=exp.platform,
            scaling_type=exp.scaling_type,
            instance_label=exp.instance_label,
            rows=exp.rows,
        )

        download_dir = os.path.join(
            config.download_dir,
            exp.platform,
            exp.scaling_type,
            exp.instance_label,
        )

        files = download_from_s3(
            bucket=config.s3_bucket,
            prefix=prefix,
            download_dir=download_dir,
        )

        if files:
            exp.local_data_dir = download_dir
            logger.info(f"Set local_data_dir for {exp.label}: {download_dir}")