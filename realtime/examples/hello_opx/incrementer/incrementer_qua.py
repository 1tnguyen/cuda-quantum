from qm import DictQuaConfig, QuantumMachinesManager
from qm.qua import *
import numpy as np

# Calculate FNV1a hash of a string. Required to populate the function-id for the rx kernel
def fnv1a_hash(s: str, bits: int = 32) -> int:
    if bits == 32:
        FNV_OFFSET_BASIS = 2166136261
        FNV_PRIME = 16777619
        MASK = 0xFFFFFFFF
    elif bits == 64:
        FNV_OFFSET_BASIS = 14695981039346656037
        FNV_PRIME = 1099511628211
        MASK = 0xFFFFFFFFFFFFFFFF
    else:
        raise ValueError("bits must be 32 or 64")
    
    hash_value = FNV_OFFSET_BASIS
    for byte in s.encode('utf-8'):
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & MASK
    
    return hash_value

# must be the same name as in the cuda kernel. In the operational mode the quake-to-qua module will handle this
device_call_function_id = fnv1a_hash("rpc_increment") 

stream_id_opx_to_opnic = 1
stream_id_opnic_to_opx = 1
RPC_MAGIC_REQUEST = 0x43555152
RPC_MAGIC_RESPONSE = 0x43555153

# Field order and widths MUST match the C++ side `RPCInputPacket` /
# `RPCOutputPacket` in `common/opnic_type.h`, which is itself byte-identical
# to the dispatcher's `RPCHeader` / `RPCResponse` (24 B header + payload).
# `request_id` increments per call; `ptp_timestamp` is a uint64 split into
# two int32 words (low, high) so QUA can carry it through unchanged.
@qua_struct
class RpcRequestPacket:
    magic: QuaArray[int, 1]
    function_id: QuaArray[int, 1]
    arg_len: QuaArray[int, 1]              # arg_len is in units of bytes
    request_id: QuaArray[int, 1]
    ptp_timestamp: QuaArray[int, 2]        # uint64 across two int32 words
    data: QuaArray[int, 1]

@qua_struct
class RpcResponsePacket:
    magic: QuaArray[int, 1]
    status: QuaArray[int, 1]
    result_len: QuaArray[int, 1]           # result_len is in units of bytes
    request_id: QuaArray[int, 1]
    ptp_timestamp: QuaArray[int, 2]
    result: QuaArray[int, 1]


import random
data_to_send = random.randint(1, 1000)

iterations = 1024

with program() as prog:

    # define two types of packets
    opx_to_opnic_packet = declare_struct(RpcRequestPacket)
    opnic_to_opx_packet = declare_struct(RpcResponsePacket)
    

    # define incoming and outgoing streams, which use the above packet structure
    opx_to_opnic_stream = declare_external_stream(RpcRequestPacket, stream_id_opx_to_opnic, QuaStreamDirection.OUTGOING)
    opnic_to_opx_stream = declare_external_stream(RpcResponsePacket, stream_id_opnic_to_opx, QuaStreamDirection.INCOMING)

    # populate the outgoing packet
    assign(opx_to_opnic_packet.magic[0], RPC_MAGIC_REQUEST)
    assign(opx_to_opnic_packet.function_id[0], device_call_function_id)
    assign(opx_to_opnic_packet.arg_len[0], 4) # we send one argument of type int, which is 4 bytes
    # PTP timestamp is unused on the OPX side; host can fill if needed.
    assign(opx_to_opnic_packet.ptp_timestamp[0], 0)
    assign(opx_to_opnic_packet.ptp_timestamp[1], 0)
    assign(opx_to_opnic_packet.data[0], data_to_send)

    latency_start = declare_stream()
    latency_end = declare_stream()
    valid = declare_stream()
    i = declare(int)
    valid_check = declare(bool)

    with for_(i, 0, i < iterations, i + 1):
        # request_id increments per call; the host echoes it in the response
        # so we can validate per-call identity end-to-end.
        assign(opx_to_opnic_packet.request_id[0], i)
        save(i, latency_start)
        # Send the packet to the opnic
        send_to_external_stream(opx_to_opnic_stream, opx_to_opnic_packet)

        # Wait for the result. This is a block call within the PPU thread
        receive_from_external_stream(opnic_to_opx_stream, opnic_to_opx_packet)
        save(i, latency_end)

        # Validate the result -- both the payload value AND the echoed
        # request_id from the response header.
        assign(valid_check,
               (opnic_to_opx_packet.result[0] == data_to_send + 1) &
               (opnic_to_opx_packet.request_id[0] == i))
        save(valid_check, valid)

    # signal shutdown to nvqlink. in this example we do it via function-id = 0
    assign(opx_to_opnic_packet.function_id[0], 0)
    send_to_external_stream(opx_to_opnic_stream, opx_to_opnic_packet)

    with stream_processing():
       latency_start.timestamps().save_all("ts_start")
       latency_end.timestamps().save_all("ts_end")
       valid.save_all("valid")

    

# qua config part
opx_ip =  "10.137.129.5" # configure your OPX ip here
opx_port = 9510 # default OPX tcp port
fem_index = 5 # configure the fem index (1-8) that you're using in your OPX

# minimalistic OPX configuration just for opnic packet commnication.
config: DictQuaConfig = {
    "controllers": {
        "con1": {
            "fems": {
                fem_index: {
                    "analog_inputs": {},
                    "analog_outputs": {1: {"offset": 0.0}},
                    "digital_inputs": {},
                    "digital_outputs": {},
                    "type": "LF"
                },
            }
        }
    },
    "digital_waveforms": {},
    "elements": {
        "qe1": {
            "intermediate_frequency": 1e6,
            "operations": {"analog": "qe1_analog_pulse"},
            "singleInput": {"port": ("con1", fem_index, 1)},
        }
    },
    "integration_weights": {},
    "mixers": {},
    "oscillators": {},
    "pulses": {
        "qe1_analog_pulse": {
            "length": 2000,
            "operation": "control",
            "waveforms": {"single": "qe1_analog_pulse_wf_I"},
        }
    },
    "waveforms": {"qe1_analog_pulse_wf_I": {"sample": 0.5, "type": "constant"}},
}

# run the program
qmm = QuantumMachinesManager(host=opx_ip, port=opx_port)
qm = qmm.open_qm(config)
job = qm.execute(prog)
job.result_handles.wait_for_all_values()

latency_start = job.result_handles.get("ts_start").fetch_all(flat_struct=True)
latency_end = job.result_handles.get("ts_end").fetch_all(flat_struct=True)
valid = job.result_handles.get("valid").fetch_all(flat_struct=True)
latency_ns = latency_end - latency_start

print("== GPU Round Trip Latency Report =====================")
print(f"Valid: {np.count_nonzero(valid)} / {len(latency_ns)}")
print(f"First: {latency_ns[0]:.2f} ns (Discarded)")
print(f"Min:    {np.min(latency_ns[1:]):.2f} ns")
print(f"Max:    {np.max(latency_ns[1:]):.2f} ns")
print(f"Avg:    {np.mean(latency_ns[1:]):.2f} ns")
print(f"Stddev: {np.std(latency_ns[1:]):.2f} ns")
print(f"95th percentile: {np.percentile(latency_ns[1:], 95):.2f} ns")
print(f"99th percentile: {np.percentile(latency_ns[1:], 99):.2f} ns")
print("======================================================")
