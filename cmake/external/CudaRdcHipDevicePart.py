#!/usr/bin/env python3
"""Write the device part of a HIP relocatable object (clang new offload driver) as assembly.

The output keeps only the offload sections of the input object: the embedded device code (.llvm.offloading), the
registration entries (llvm_offload_entries) and their names (.llvm.rodata.offloading). The host symbols the entries
point to (kernel handles, host shadows of device variables) become undefined references, so an object assembled from
the output carries the device code and its registration without any host code: linked into a shared library with
`-fgpu-rdc --hip-link`, it registers the kernels of host code living in another shared library.

Usage:
  CudaRdcHipDevicePart.py <object.o> <output.s>
      writes the device part of one object as assembly (no file if the object has no device code)
  CudaRdcHipDevicePart.py --archive <in.a> <out.a> <compiler> <ar>
      writes an archive of the device parts of the members of a static archive
"""

import os
import shutil
import struct
import subprocess
import sys
import tempfile

NAMES, ENTRIES, OFFLOADING = (
    ".llvm.rodata.offloading",
    "llvm_offload_entries",
    ".llvm.offloading",
)
R_X86_64_64 = 1
# Registration entry layout (llvm::offloading::EntryTy, version 1): uint64 reserved; uint16 version; uint16 kind;
# uint32 flags; address at +16; name at +24; size at +32 (0 for kernels); data; auxiliary address: 56 bytes.
# Supported (version, kind) pairs; kind 4 is HIP. A compiler writing another layout must be added here explicitly.
ENTRY_SIZE, ENTRY_VERSION, ENTRY_ADDRESS, ENTRY_SIZE_FIELD = 56, 8, 16, 32
SUPPORTED_ENTRIES = {(1, 4)}


def unsupported(path, what):
    sys.exit(
        f"{path}: {what}. CudaRdcHipDevicePart.py supports the clang new offload driver with registration entries "
        f"version/kind {sorted(SUPPORTED_ENTRIES)}; this compiler needs an update of the script"
    )


def main(path, out):
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        sys.exit(f"{path}: not a little-endian ELF64 object")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = []
    for i in range(shnum):
        name, stype, flags, addr, off, size, link, info, align, entsize = (
            struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        )
        sections.append(
            dict(
                name=name,
                type=stype,
                off=off,
                size=size,
                link=link,
                info=info,
                align=align,
                entsize=entsize,
            )
        )
    shstr = sections[shstrndx]

    def cstr(off):
        return data[off : data.index(b"\0", off)].decode()

    for s in sections:
        s["label"] = cstr(shstr["off"] + s["name"])
    index = {s["label"]: i for i, s in enumerate(sections)}
    if any(label.startswith("__CLANG_OFFLOAD_BUNDLE__") for label in index):
        unsupported(
            path,
            "object compiled with the old offload driver (offload bundles); compile with "
            "--offload-new-driver",
        )
    if OFFLOADING not in index:
        return  # host-only object
    other_entries = [
        label
        for label in index
        if "offload" in label
        and "entries" in label
        and label != ENTRIES
        and not label.startswith(".rela")
    ]
    if other_entries:
        unsupported(path, f"unknown registration entry sections {other_entries}")
    has_entries = (
        ENTRIES in index and NAMES in index
    )  # absent when the object has no kernels or device variables
    symtab = next(s for s in sections if s["type"] == 2)
    strtab = sections[symtab["link"]]
    symbols = []
    for i in range(symtab["size"] // 24):
        name, info, other, shndx, value, size = struct.unpack_from(
            "<IBBHQQ", data, symtab["off"] + 24 * i
        )
        symbols.append(
            dict(
                name=cstr(strtab["off"] + name),
                bind=info >> 4,
                type=info & 0xF,
                shndx=shndx,
                value=value,
            )
        )
    rela = next(
        (
            s
            for s in sections
            if has_entries and s["type"] == 4 and s["info"] == index[ENTRIES]
        ),
        None,
    )
    relocs = {}
    if rela:
        for i in range(rela["size"] // 24):
            off, info, addend = struct.unpack_from("<QQq", data, rela["off"] + 24 * i)
            if info & 0xFFFFFFFF != R_X86_64_64:
                sys.exit(
                    f"{path}: unexpected relocation type {info & 0xFFFFFFFF} in {ENTRIES}"
                )
            relocs[off] = (symbols[info >> 32], addend)

    dev_sec = sections[index[OFFLOADING]]
    tag = "dp_" + str(abs(hash(path)) % 10**12)

    def target(sym, addend):
        # Entry names are local strings of a mergeable section: point at a label on the referenced string itself,
        # since the linker merges the strings and an offset past the start of a string would not follow it
        if sym["shndx"] == index[NAMES]:
            return f".L{tag}_name{(sym['value'] if sym['type'] != 3 else 0) + addend}"
        return sym["name"] + (
            f"+{addend}" if addend > 0 else (f"{addend}" if addend < 0 else "")
        )

    lines = ["# Device part of " + path, ""]
    if has_entries:
        lines += entry_lines(path, data, sections, index, symbols, relocs, tag, target)
    devfile = out + ".offloading.bin"
    open(devfile, "wb").write(data[dev_sec["off"] : dev_sec["off"] + dev_sec["size"]])
    lines += [
        "",
        f'.section {OFFLOADING},"e",@llvm_offloading',
        f".p2align {max(dev_sec['align'], 1).bit_length() - 1}",
        f'.incbin "{devfile}"',
        "",
    ]
    open(out, "w").write("\n".join(lines))


def entry_lines(path, data, sections, index, symbols, relocs, tag, target):
    names_sec, entries_sec = sections[index[NAMES]], sections[index[ENTRIES]]
    lines = [f'.section {NAMES},"aMS",@progbits,1']
    blob = data[names_sec["off"] : names_sec["off"] + names_sec["size"]]
    start = 0
    for end, byte in enumerate(blob):
        if byte == 0:
            lines += [
                f".L{tag}_name{start}:",
                '.asciz "' + blob[start:end].decode() + '"',
            ]
            start = end + 1
    lines += [
        "",
        f'.section {ENTRIES},"aw",@progbits',
        f".p2align {max(entries_sec['align'], 1).bit_length() - 1}",
    ]
    weak = {
        s["value"]: s["name"]
        for s in symbols
        if s["shndx"] == index[ENTRIES] and s["bind"] == 2
    }
    blob = data[entries_sec["off"] : entries_sec["off"] + entries_sec["size"]]
    if len(blob) % ENTRY_SIZE:
        sys.exit(
            f"{path}: {ENTRIES} size {len(blob)} is not a multiple of {ENTRY_SIZE}"
        )
    for start in range(0, len(blob), ENTRY_SIZE):
        version, kind = struct.unpack_from("<HH", blob, start + ENTRY_VERSION)
        if (version, kind) not in SUPPORTED_ENTRIES:
            unsupported(path, f"registration entry version {version}, kind {kind}")
        # A host symbol with internal linkage is invisible outside its host library: a kernel launched through it
        # cannot be registered from a separate final library; a variable's entry only serves host symbol access
        sym, addend = relocs.get(start + ENTRY_ADDRESS, (None, 0))
        if sym and sym["bind"] == 0:
            name = local_name(symbols, sym, addend)
            (size,) = struct.unpack_from("<Q", blob, start + ENTRY_SIZE_FIELD)
            if size == 0:
                sys.exit(
                    f"{path}: kernel {name} has internal linkage (static or in an anonymous namespace); "
                    "give it external linkage (e.g. inline) to launch it from a shared library"
                )
            print(
                f"{path}: warning: device variable {name} has internal linkage; its host-side symbol access is "
                "not registered",
                file=sys.stderr,
            )
            continue
        for pos in range(start, start + ENTRY_SIZE):
            if pos in weak:
                lines += [f".weak {weak[pos]}", f"{weak[pos]}:"]
        pos = start
        while pos < start + ENTRY_SIZE:
            if pos in relocs:
                lines.append(".quad " + target(*relocs[pos]))
                pos += 8
            else:
                lines.append(f".byte {blob[pos]}")
                pos += 1
    return lines


def local_name(symbols, sym, addend):
    """Name of the local symbol a relocation points to (section symbol + addend, or the symbol itself)"""
    if sym["type"] != 3:
        return sym["name"]
    match = [
        s["name"]
        for s in symbols
        if s["shndx"] == sym["shndx"] and s["value"] == addend and s["type"] != 3
    ]
    return match[0] if match else f"<section {sym['shndx']} + {addend}>"


def archive_members(path):
    """(name, bytes) of the members of a System V / GNU ar archive"""
    data = open(path, "rb").read()
    if data[:8] != b"!<arch>\n":
        sys.exit(f"{path}: not an ar archive")
    pos, longnames, members = 8, b"", []
    while pos + 60 <= len(data):
        header = data[pos : pos + 60]
        name, size = header[:16].decode().rstrip(), int(header[48:58].decode())
        body = data[pos + 60 : pos + 60 + size]
        pos += 60 + size + (size & 1)
        if name == "//":
            longnames = body
        elif name in ("/", "/SYM64/"):
            continue
        else:
            if name.startswith("/") and name[1:].isdigit():
                off = int(name[1:])
                name = longnames[off : longnames.index(b"\n", off)].decode()
            members.append((name.rstrip("/"), body))
    return members


def archive(src, out, compiler, ar):
    work = tempfile.mkdtemp(prefix="hipdev_")
    try:
        objects = []
        for i, (name, body) in enumerate(archive_members(src)):
            obj = os.path.join(work, f"{i}_{os.path.basename(name)}")
            open(obj, "wb").write(body)
            asm = obj + ".dev.s"
            main(obj, asm)
            if os.path.exists(asm):
                subprocess.run([compiler, "-c", asm, "-o", obj + ".dev.o"], check=True)
                objects.append(obj + ".dev.o")
        if os.path.exists(out):
            os.remove(out)
        # An empty archive is valid: the library had no device code
        subprocess.run([ar, "qcs", out] + objects, check=True)
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    if sys.argv[1] == "--archive":
        archive(*sys.argv[2:6])
    else:
        main(*sys.argv[1:3])
