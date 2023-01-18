# From: https://github.com/facebookresearch/co3d/blob/main/co3d/download_dataset.py.
# python3 download_dataset.py --download_folder DOWNLOAD_FOLDER --single_sequence_subset

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import functools
import hashlib
import json
import os
import requests
import shutil

from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Optional

DEFAULT_LINK_LIST_FILE = os.path.join(os.path.dirname(__file__), "links.json")
SHA256S_FILE = os.path.join(os.path.dirname(__file__), "co3d_sha256.json")
BLOCKSIZE = 65536


def get_expected_sha256s(single_sequence_subset: bool = False):
    with open(SHA256S_FILE, "r") as f:
        expected_sha256s = json.load(f)
    if single_sequence_subset:
        return expected_sha256s["singlesequence"]
    else:
        return expected_sha256s["full"]


def sha256_file(path: str):
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        file_buffer = f.read(BLOCKSIZE)
        while len(file_buffer) > 0:
            sha256_hash.update(file_buffer)
            file_buffer = f.read(BLOCKSIZE)
    digest_ = sha256_hash.hexdigest()
    # print(f"{digest_} {path}")
    return digest_


def check_co3d_sha256(
    path: str,
    expected_sha256s: Optional[dict] = None,
    single_sequence_subset: bool = False,
):
    zipname = os.path.split(path)[-1]
    if expected_sha256s is None:
        expected_sha256s = get_expected_sha256s(single_sequence_subset)
    extracted_hash = sha256_file(path)
    assert (
        extracted_hash == expected_sha256s[zipname]
    ), f"{extracted_hash} != {expected_sha256s[zipname]}"


def main(
    link_list_file: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    download_categories: Optional[List[str]] = None,
    checksum_check: bool = False,
    single_sequence_subset: bool = False,
    clear_archives_after_unpacking: bool = False,
):
    """
    Downloads and unpacks the CO3D dataset.

    Args:
        link_list_file: A text file with the list of CO3D file download links.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers
            for extracting the dataset files.
        download_categories: A list of categories to download.
            If `None`, downloads all.
        checksum_check: Enable validation of the downloaded file's checksum before
            extraction.
        single_sequence_subset: Whether the downloaded dataset is the single-sequence
            subset of the full dataset.
        clear_archives_after_unpacking: Delete the unnecessary downloaded archive files
            after unpacking.
    """

    if not os.path.isfile(link_list_file):
        raise ValueError(
            "Please specify `link_list_file` with a valid path to a file"
            " with CO3D download links."
            " The file is stored in the co3d github:"
            " https://https://github.com/facebookresearch/co3d/blob/main/co3d/links.txt"
        )

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the CO3D dataset."
            + f" {download_folder} does not exist."
        )

    # read the link file
    with open(link_list_file, "r") as f:
        links = json.load(f)

    # get the full dataset links or the single-sequence subset links
    links = links["singlesequence"] if single_sequence_subset else links["full"]

    # split to data links and the links containing json metadata
    metadata_links = []
    data_links = []
    for category_name, urls in links.items():
        for url in urls:
            link_name = os.path.split(url)[-1]
            if single_sequence_subset:
                link_name = link_name.replace("_singlesequence", "")
            if category_name == "METADATA":
                metadata_links.append((link_name, url))
            else:
                data_links.append((category_name, link_name, url))

    if download_categories is not None:
        co3d_categories = set(l[0] for l in data_links)
        not_in_co3d = [c for c in download_categories if c not in co3d_categories]
        if len(not_in_co3d) > 0:
            raise ValueError(
                f"download_categories {str(not_in_co3d)} are not valid"
                + "CO3D categories."
            )
        data_links = [(c, ln, l) for c, ln, l in data_links if c in download_categories]

    with Pool(processes=n_download_workers) as download_pool:
        print(f"Downloading {len(metadata_links)} CO3D metadata files ...")
        for _ in tqdm(
            download_pool.imap(
                functools.partial(_download_metadata_file, download_folder),
                metadata_links,
            ),
            total=len(metadata_links),
        ):
            pass

        print(f"Downloading {len(data_links)} CO3D dataset files ...")
        for _ in tqdm(
            download_pool.imap(
                functools.partial(_download_category_file, download_folder),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print(f"Extracting {len(data_links)} CO3D dataset files ...")
    with Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_category_file,
                    download_folder,
                    checksum_check,
                    single_sequence_subset,
                    clear_archives_after_unpacking,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print("Done")


def _unpack_category_file(
    download_folder: str,
    checksum_check: bool,
    single_sequence_subset: bool,
    clear_archive: bool,
    link: str,
):
    category, link_name, url = link
    local_fl = os.path.join(download_folder, link_name)
    if checksum_check:
        print(f"Checking SHA256 for {local_fl}.")
        check_co3d_sha256(local_fl, single_sequence_subset=single_sequence_subset)
    print(f"Unpacking CO3D dataset file {local_fl} ({link_name}) to {download_folder}.")
    shutil.unpack_archive(local_fl, download_folder)
    if clear_archive:
        os.remove(local_fl)


def _download_category_file(download_folder: str, link: str):
    category, link_name, url = link
    local_fl = os.path.join(download_folder, link_name)
    print(f"Downloading CO3D dataset file {link_name} ({url}) to {local_fl}.")
    _download_with_progress_bar(url, local_fl, link_name)


def _download_metadata_file(download_folder: str, link: str):
    local_fl = os.path.join(download_folder, link[0])
    # remove the singlesequence postfix in case we are downloading the s.s. subset
    local_fl = local_fl.replace("_singlesequence", "")
    print(f"Downloading CO3D metadata file {link[1]} ({link[0]}) to {local_fl}.")
    _download_with_progress_bar(link[1], local_fl, link[0])


def _download_with_progress_bar(url: str, fname: str, filename: str):
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    print(url)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for datai, data in enumerate(resp.iter_content(chunk_size=1024)):
            size = file.write(data)
            bar.update(size)
            if datai % max((max(total // 1024, 1) // 20), 1) == 0:
                print(
                    f"{filename}: Downloaded {100.0 * (float(bar.n) / max(total, 1)):3.1f}%."
                )
                print(bar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the CO3D dataset.")
    parser.add_argument(
        "--download_folder",
        type=str,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--n_download_workers",
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        "--n_extract_workers",
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        "--download_categories",
        type=lambda x: [x_.strip() for x_ in x.split(",")],
        default=None,
        help="A comma-separated list of CO3D categories to download."
        + " Example: 'orange,car' will download only oranges and cars",
    )
    parser.add_argument(
        "--link_list_file",
        type=str,
        default=DEFAULT_LINK_LIST_FILE,
        help=(
            "The file with html links to the CO3D dataset files."
            + " In most cases the default local file `links.json` should be used."
        ),
    )
    parser.add_argument(
        "--checksum_check",
        action="store_true",
        default=False,
        help="Check the SHA256 checksum of each downloaded file before extraction.",
    )
    parser.add_argument(
        "--single_sequence_subset",
        action="store_true",
        default=False,
        help="Download the single-sequence subset of the dataset.",
    )
    parser.add_argument(
        "--clear_archives_after_unpacking",
        action="store_true",
        default=False,
        help="Delete the unnecessary downloaded archive files after unpacking.",
    )
    args = parser.parse_args()
    main(
        str(args.link_list_file),
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        download_categories=args.download_categories,
        checksum_check=bool(args.checksum_check),
        single_sequence_subset=bool(args.single_sequence_subset),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
    )
