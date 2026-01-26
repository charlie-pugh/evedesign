import random
import time
from io import StringIO

from loguru import logger

from protdesign.tools.api_utils import _request_with_retries
from protdesign.structure import Structure
from protdesign.__about__ import __version__


def _clean_sequence(seq):
    return "".join(seq.split()).upper()


def _predict_3di(sequence, host_url, headers):
    res = _request_with_retries(
        "GET",
        f"{host_url}/predict/{sequence}",
        headers=headers,
        context="Foldseek 3Di server",
    )
    text = res.text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    text = text.replace('"', "").replace("'", "")
    return "".join(text.split())


def _build_3di_query(sequence, header, host_url, headers):
    seq = _clean_sequence(sequence)
    if not seq:
        raise ValueError("Empty sequence for Foldseek 3Di prediction")
    three_di = _predict_3di(seq, host_url, headers)
    return f">{header}\n{seq}\n>3DI\n{three_di}\n"


def _foldseek_submit(query_text, databases, mode, host_url, headers):
    payload = [("q", query_text), ("mode", mode)]
    for db in databases:
        payload.append(("database[]", db))
    res = _request_with_retries(
        "POST",
        f"{host_url}/api/ticket",
        data=payload,
        headers=headers,
        context="Foldseek server",
    )
    try:
        return res.json()
    except ValueError:
        logger.error(f"Server didn't reply with json: {res.text}")
        return {"status": "ERROR"}


def _foldseek_status(ticket, host_url, headers):
    res = _request_with_retries(
        "GET",
        f"{host_url}/api/ticket/{ticket}",
        headers=headers,
        context="Foldseek server",
    )
    try:
        return res.json()
    except ValueError:
        logger.error(f"Server didn't reply with json: {res.text}")
        return {"status": "ERROR"}


def _foldseek_result(ticket, entry, host_url, headers, params=None):
    res = _request_with_retries(
        "GET",
        f"{host_url}/api/result/{ticket}/{entry}",
        params=params,
        headers=headers,
        context="Foldseek server",
    )
    try:
        return res.json()
    except ValueError:
        return res.text


def _extract_hits_brief(result_obj):
    results = []
    if isinstance(result_obj, list):
        if all(isinstance(item, dict) for item in result_obj):
            return result_obj
        return results

    if isinstance(result_obj, dict):
        results_list = result_obj.get("results", [])
        if not isinstance(results_list, list):
            return results
        for result in results_list:
            if not isinstance(result, dict):
                continue
            alignments = result.get("alignments", [])
            if not isinstance(alignments, list):
                continue
            for alignment_group in alignments:
                if not isinstance(alignment_group, list):
                    continue
                for hit in alignment_group:
                    if isinstance(hit, dict):
                        results.append(hit)
        return results

    return results


def foldseek_search_sequence(
    sequence,
    databases: list[str] = ["pdb100"],
    mode: str = "3diaa",
    host_url: str = "https://search.foldseek.com",
    predict_host_url: str = "https://3di.foldseek.com",
    user_agent: str | None = None,
):
    """
    Submit a single AA sequence to Foldseek and return brief hits without full C-alpha coords.
    Use foldseek_fetch_full_hit to retrieve full hit data including C-alpha coordinates.
    """
    if user_agent is None:
        user_agent = "evedesign/" + __version__

    headers = {"User-Agent": user_agent} if user_agent else None

    query_text = _build_3di_query(
        sequence,
        header="query",
        host_url=predict_host_url,
        headers=headers,
    )

    out = _foldseek_submit(query_text, databases, mode, host_url, headers)
    while out.get("status") in ["UNKNOWN", "RATELIMIT"]:
        sleep_time = 5 + random.randint(0, 5)
        logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
        time.sleep(sleep_time)
        out = _foldseek_submit(query_text, databases, mode, host_url, headers)

    if out.get("status") == "ERROR":
        raise Exception(
            "Foldseek API is giving errors. Please confirm your query is valid. "
            "If error persists, please try again an hour later."
        )

    if out.get("status") == "MAINTENANCE":
        raise Exception(
            "Foldseek API is undergoing maintenance. Please try again in a few minutes."
        )

    ticket = out.get("id")
    if not ticket:
        raise RuntimeError(f"Foldseek did not return a ticket id: {out}")

    out = {"status": "UNKNOWN"}
    while out.get("status") in ["UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"]:
        t = 5 + random.randint(0, 5)
        logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
        time.sleep(t)
        out = _foldseek_status(ticket, host_url, headers)

    if out.get("status") == "MAINTENANCE":
        raise Exception(
            "Foldseek API is undergoing maintenance. Please try again in a few minutes."
        )

    if out.get("status") == "ERROR":
        raise Exception(
            "Foldseek API is giving errors. Please confirm your query is valid. "
            "If error persists, please try again an hour later."
        )

    if out.get("status") != "COMPLETE":
        raise RuntimeError(f"Unexpected Foldseek status: {out.get('status')}")

    result_obj = _foldseek_result(
        ticket,
        entry=0,
        host_url=host_url,
        headers=headers,
        params={"format": "brief"},
    )
    hits = _extract_hits_brief(result_obj)
    return hits, ticket


def foldseek_fetch_full_hit(
    ticket,
    index,
    database,
    entry: int = 0,
    host_url: str = "https://search.foldseek.com",
    user_agent: str | None = None,
):
    """
    Fetch full hit data using index+database (format=brief).
    """
    if user_agent is None:
        user_agent = "evedesign/" + __version__

    headers = {"User-Agent": user_agent} if user_agent else None

    return _foldseek_result(
        ticket,
        entry=entry,
        host_url=host_url,
        headers=headers,
        params={"format": "brief", "index": index, "database": database},
    )



AA1_TO_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "B": "ASX",
    "Z": "GLX",
    "J": "XLE",
    "U": "SEC",
    "O": "PYL",
    "X": "UNK",
    "-": "UNK",
}


def _parse_ca_coords(tca):
    if tca is None:
        return []
    if isinstance(tca, str):
        parts = [p for p in tca.split(",") if p.strip()]
        try:
            coords = [float(p) for p in parts]
        except ValueError:
            return []
    elif isinstance(tca, (list, tuple)):
        try:
            coords = [float(p) for p in tca]
        except (TypeError, ValueError):
            return []
    else:
        return []

    if len(coords) < 3:
        return []
    if len(coords) % 3 != 0:
        coords = coords[: len(coords) - (len(coords) % 3)]
    return [coords[i:i + 3] for i in range(0, len(coords), 3)]


def _mock_pdb_from_ca(tca, seq, chain_id="A"):
    coords = _parse_ca_coords(tca)
    if not coords:
        return ""
    chain_id = (chain_id or "A")[:1]
    seq = _clean_sequence(seq) if seq else ""
    use_seq = len(seq) == len(coords)
    lines = []
    for idx, (x, y, z) in enumerate(coords, start=1):
        aa = seq[idx - 1] if use_seq else "A"
        res = AA1_TO_AA3.get(aa, "UNK")
        lines.append(
            f"ATOM  {idx:5d}  CA  {res:>3} {chain_id:1}{idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    return "\n".join(lines)


def build_structure_from_ca(tca, seq, chain_id="A"):
    pdb_text = _mock_pdb_from_ca(tca, seq, chain_id=chain_id)
    if not pdb_text:
        return ""
    return Structure(StringIO(pdb_text), format="pdb")


def build_structure_from_hit(hit, chain_id="A"):
    if not isinstance(hit, dict):
        return ""
    return build_structure_from_ca(hit.get("tCa"), hit.get("tSeq"), chain_id=chain_id)


def flatten_foldseek_hits(hit_groups):
    flattened = []
    for group in hit_groups:
        if isinstance(group, list):
            flattened.extend([hit for hit in group if isinstance(hit, dict)])
        elif isinstance(group, dict):
            flattened.append(group)
    return flattened


def hits_to_structures(hits, chain_id="A"):
    paths = {}
    for idx, hit in enumerate(hits):
        mmcif = build_structure_from_hit(hit, chain_id=chain_id)
        if not mmcif:
            continue
        paths[idx] = mmcif
    return paths
