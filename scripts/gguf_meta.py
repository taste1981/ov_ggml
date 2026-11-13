"""Print GGUF metadata without numpy: selected keys, and vocabulary entries by id.

usage: python -I gguf_meta.py <model.gguf> [key-substring ...] [--tokens id,id,...] [--find text,...]
                              [--largest N]
With no key substrings, prints every scalar key. Token types: 1 normal, 2 unknown, 3 control,
4 user-defined, 5 unused, 6 byte.
"""
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def read_value(f, vtype):
    if vtype in SCALAR:
        fmt = SCALAR[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 8:
        return read_str(f)
    if vtype == 9:
        etype, count = struct.unpack("<IQ", f.read(12))
        return [read_value(f, etype) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {vtype}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[2:]
    token_ids, find = [], []
    if "--tokens" in args:
        i = args.index("--tokens")
        token_ids = [int(x) for x in args[i + 1].split(",")]
        args = args[:i] + args[i + 2:]
    if "--find" in args:  # exact token texts, comma-separated
        i = args.index("--find")
        find = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    largest = 0
    if "--largest" in args:  # list the N largest tensors
        i = args.index("--largest")
        largest = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    with open(sys.argv[1], "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        assert magic == b"GGUF", magic
        meta = {}
        for _ in range(n_kv):
            key = read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            meta[key] = read_value(f, vtype)
        tensors = []
        for _ in range(n_tensors if largest else 0):
            name = read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            ttype, offset = struct.unpack("<IQ", f.read(12))
            tensors.append((offset, name, dims, ttype))
        if largest:
            align = meta.get("general.alignment", 32)
            data_start = (f.tell() + align - 1) // align * align
            end = f.seek(0, 2) - data_start
    if largest:  # a tensor's size is the distance to the next one's offset
        tensors.sort()
        sized = [(nxt - off, name, dims, ttype) for (off, name, dims, ttype), nxt in
                 zip(tensors, [t[0] for t in tensors[1:]] + [end])]
        for size, name, dims, ttype in sorted(sized, reverse=True)[:largest]:
            print(f"tensor {name}: {size / 2**20:9.1f} MiB  dims={list(dims)}  ggml_type={ttype}")
    print(f"GGUF v{version}, {n_tensors} tensors, {n_kv} keys")
    for k, v in meta.items():
        if isinstance(v, list):
            if args and any(a in k for a in args):
                print(f"{k} = [{len(v)} items] {v[:8]}{' ...' if len(v) > 8 else ''}")
            continue
        if not args or any(a in k for a in args):
            text = v if not isinstance(v, str) or len(v) < 300 else v[:300] + f"... ({len(v)} chars)"
            print(f"{k} = {text!r}")
    tokens, types = meta.get("tokenizer.ggml.tokens", []), meta.get("tokenizer.ggml.token_type", [])
    token_ids += [i for i, t in enumerate(tokens) if t in find]
    for i in token_ids:
        if i >= len(tokens):
            print(f"token {i}: beyond the {len(tokens)}-entry vocabulary")
            continue
        print(f"token {i}: {tokens[i]!r} type={types[i] if i < len(types) else '?'}")


if __name__ == "__main__":
    main()
