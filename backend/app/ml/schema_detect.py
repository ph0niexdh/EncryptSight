import pandas as pd
from typing import List, Tuple, Dict, Any

# 42 raw feature columns expected for UNSW-NB15 lane
UNSW_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb",
    "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports"
]

# Core feature columns expected for CICIDS2017 lane (normalized by stripping whitespace)
CICIDS_FEATURES_REQUIRED = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", 
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", 
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", 
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", 
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", 
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", 
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", 
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", 
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", 
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", 
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", 
    "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", 
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", 
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

def clean_column_names(columns: List[str]) -> List[str]:
    """Strips leading/trailing whitespace from column names."""
    return [col.strip() if isinstance(col, str) else col for col in columns]

def detect_schema(filepath_or_buffer: Any) -> str:
    """
    Detects whether the CSV file belongs to the UNSW-NB15 schema or the CICIDS2017 schema.
    Reads only the header row for speed.
    """
    # Read just the header
    df_header = pd.read_csv(filepath_or_buffer, nrows=0)
    columns_raw = list(df_header.columns)
    columns = clean_column_names(columns_raw)
    columns_lower = [col.lower() for col in columns]

    # Check for UNSW indicators: specific lowercase features
    unsw_indicators = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes"]
    has_unsw = all(ind in columns_lower for ind in unsw_indicators)

    # Check for CICIDS indicators: specific spaces and CamelCase features
    cicids_indicators = ["flow duration", "total fwd packets", "total backward packets", "flow bytes/s"]
    has_cicids = any(ind in columns_lower for ind in cicids_indicators)

    if has_unsw:
        return "unsw"
    elif has_cicids:
        return "cicids"
    else:
        raise ValueError(
            "Could not identify CSV schema. Ensure the file follows the UNSW-NB15 schema "
            "(with columns like dur, proto, service, state) or the CICIDS2017 schema "
            "(with columns like Flow Duration, Total Fwd Packets, Flow Bytes/s)."
        )

def validate_schema(filepath_or_buffer: Any, schema_type: str) -> Tuple[bool, List[str]]:
    """
    Validates that the file has all required features for the specified schema type.
    Returns (is_valid, list_of_missing_columns).
    """
    df_header = pd.read_csv(filepath_or_buffer, nrows=0)
    columns = clean_column_names(list(df_header.columns))

    if schema_type == "unsw":
        # Convert all to lowercase for case-insensitive validation
        columns_lower = [col.lower() for col in columns]
        missing = [feat for feat in UNSW_FEATURES if feat.lower() not in columns_lower]
        return len(missing) == 0, missing

    elif schema_type == "cicids":
        # Check against normalized CICIDS features
        # Note: We compare case-insensitively to be robust
        columns_lower = [col.lower() for col in columns]
        missing = [feat for feat in CICIDS_FEATURES_REQUIRED if feat.lower() not in columns_lower]
        return len(missing) == 0, missing

    else:
        raise ValueError(f"Unknown schema type: {schema_type}")
