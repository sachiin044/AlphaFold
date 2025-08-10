# test_infer.py
from src.models.infer_wrapper import predict_by_domain_id, predict_sequence

# ---- Test 1: Using precomputed embedding via domain_id ----
print("Domain test ->", predict_by_domain_id("d3i3eb3"))

# ---- Test 2: Using raw sequence ----
seq = "gdtrprflwqlkfechffngtervrllersiynqeesvrfdsdvgeyravtelgrpdaeywnsqkdlleqrraavdtycrhnygvgesftvq".upper()
try:
    print("Sequence test ->", predict_sequence(seq, use_esm_on_the_fly=False))
except Exception as e:
    print("Sequence prediction failed:", e)
