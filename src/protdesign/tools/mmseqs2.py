import io
import os
import random
import tarfile
import tempfile
import time

from pathlib import Path
from loguru import logger

from protdesign.__about__ import __version__
from protdesign.tools.api_utils import _request_with_retries
from protdesign.entity import System
from protdesign.sequence import read_fasta, Sequence, Sequences

def run_mmseqs2(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter=None,
    use_pairing=False,
    pairing_strategy="greedy",
    host_url="https://api.colabfold.com",
    user_agent: str = "",
):
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent (e.g., 'toolname/version contact@email') "
            "to help us debug in case of problems. This warning will become an error in the future."
        )

    if use_templates:
        logger.warning("Template fetching disabled; proceeding without templates.")

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        res = _request_with_retries(
            "POST",
            f"{host_url}/{submission_endpoint}",
            data={"q": query, "mode": mode},
            timeout=6.02,
            headers=headers,
            context="MSA server",
        )
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        res = _request_with_retries(
            "GET",
            f"{host_url}/ticket/{ID}",
            timeout=6.02,
            headers=headers,
            context="MSA server",
        )
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        res = _request_with_retries(
            "GET",
            f"{host_url}/result/download/{ID}",
            timeout=6.02,
            headers=headers,
            context="MSA server",
        )
        with open(path, "wb") as out:
            out.write(res.content)

    seqs = [x] if isinstance(x, str) else x

    if filter is not None:
        use_filter = filter

    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = f"{mode}-env"

    path = f"{prefix}_{mode}"
    os.makedirs(path, exist_ok=True)

    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    seqs_unique = []
    for seq in seqs:
        if seq not in seqs_unique:
            seqs_unique.append(seq)
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    if not os.path.isfile(tar_gz_file):
        while REDO:
            out = submit(seqs_unique, mode, N)
            while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                sleep_time = 5 + random.randint(0, 5)
                logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                time.sleep(sleep_time)
                out = submit(seqs_unique, mode, N)

            if out["status"] == "ERROR":
                raise Exception(
                    "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. "
                    "If error persists, please try again an hour later."
                )

            if out["status"] == "MAINTENANCE":
                raise Exception(
                    "MMseqs2 API is undergoing maintenance. Please try again in a few minutes."
                )

            ID = out["id"]
            while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                t = 5 + random.randint(0, 5)
                logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                time.sleep(t)
                out = status(ID)

            if out["status"] == "COMPLETE":
                REDO = False

            if out["status"] == "ERROR":
                REDO = False
                raise Exception(
                    "MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. "
                    "If error persists, please try again an hour later."
                )

        download(ID, tar_gz_file)

    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path, filter="data")

    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        with open(a3m_file, "r") as handle:
            for line in handle:
                if line:
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    return ["".join(a3m_lines[n]) for n in Ms]


def _parse_a3m(a3m_text):
    return list(read_fasta(io.StringIO(a3m_text)))


def _sequences_from_entries(entries, keys=None):
    seqs = []
    for i, (header, seq) in enumerate(entries):
        key = None
        if keys is not None and i < len(keys):
            key = keys[i]
        seqs.append(Sequence(seq=seq, id=header.split()[0], key=key))
    return Sequences(seqs, aligned=True, format="a3m")


def add_sequences_mmseqs2(
    system: System,
    use_env: bool = False,
    use_filter: bool = True,
    filter=None,
    use_pairing: bool = False,
    pair_mode: str = "unpaired_paired",
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    user_agent: str | None = None,
    keep_tmp_dir: bool = False,
    tmpdir: str | Path | None = None,
) -> System:
    """
    Attach MSAs to all protein entities in system.
    """
    protein_entity_reps = [
        (idx, "".join(entity.rep))
        for idx, entity in enumerate(system)
        if entity.type_ == "protein" and entity.rep is not None
    ]
    if not protein_entity_reps:
        return system.copy()

    query_seqs_unique = []
    for _, seq in protein_entity_reps:
        if seq not in query_seqs_unique:
            query_seqs_unique.append(seq)

    if user_agent is None:
        user_agent = "evedesign/" + __version__

    if not use_pairing:
        pair_mode = "unpaired"
    else:
        pair_mode = pair_mode.lower()
        if pair_mode not in {"paired", "unpaired", "unpaired_paired"}:
            raise ValueError(f"Invalid pair_mode: {pair_mode}")

    need_unpaired = pair_mode in {"unpaired", "unpaired_paired"}
    need_paired = pair_mode in {"paired", "unpaired_paired"}

    tmpdir_ctx = None
    if keep_tmp_dir:
        tmpdir_path = Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp(prefix="mmseqs_"))
        tmpdir_path.mkdir(parents=True, exist_ok=True)
        logger.info("Keeping MMseqs2 output in %s", tmpdir_path)
    else:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir_path = Path(tmpdir_ctx.name)

    try:
        unpaired_a3m_lines = None
        if need_unpaired:
            unpaired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                tmpdir_path.joinpath("mmseqs_out"),
                use_env=use_env,
                use_filter=use_filter,
                use_templates=False,
                filter=filter,
                use_pairing=False,
                pairing_strategy=pairing_strategy,
                host_url=host_url,
                user_agent=user_agent,
            )

        paired_a3m_lines = None
        if need_paired and len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                tmpdir_path.joinpath("mmseqs_out"),
                use_env=use_env,
                use_filter=use_filter,
                use_templates=False,
                filter=filter,
                use_pairing=True,
                pairing_strategy=pairing_strategy,
                host_url=host_url,
                user_agent=user_agent,
            )
    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.cleanup()

    unpaired_entries_by_seq = {}
    if unpaired_a3m_lines is not None:
        unpaired_entries_by_seq = {
            seq: _parse_a3m(a3m_text)
            for seq, a3m_text in zip(query_seqs_unique, unpaired_a3m_lines)
        }

    paired_entries_by_seq = {}
    paired_keys = None
    if paired_a3m_lines is not None:
        paired_entries_by_seq = {
            seq: _parse_a3m(a3m_text)
            for seq, a3m_text in zip(query_seqs_unique, paired_a3m_lines)
        }
        paired_lengths = [len(entries) for entries in paired_entries_by_seq.values()]
        if paired_lengths:
            paired_len = min(paired_lengths)
            if any(length != paired_len for length in paired_lengths):
                logger.warning(
                    "Paired MSA lengths differ across chains; assigning keys to the first %d rows.",
                    paired_len,
                )
            paired_keys = [f"pair-{i}" for i in range(paired_len)]

    system = system.copy()
    for entity_idx, rep in protein_entity_reps:
        paired_entries = paired_entries_by_seq.get(rep, [])
        unpaired_entries = unpaired_entries_by_seq.get(rep, [])

        seqs = []
        if paired_entries:
            seqs.extend(
                _sequences_from_entries(paired_entries, keys=paired_keys).seqs
            )
        if unpaired_entries:
            seqs.extend(_sequences_from_entries(unpaired_entries).seqs)

        if seqs:
            system[entity_idx].sequences = Sequences(seqs, aligned=True, format="a3m")

    return system

