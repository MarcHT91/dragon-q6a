#!/bin/sh
# Build a Module.symvers (no CRCs: CONFIG_MODVERSIONS is off) from the installed
# modules of kernel $1, so modpost records correct "depends:" for our module.
kver="$1"; out="$2"
: > "$out"
find "/lib/modules/$kver/kernel" -name '*.ko' | while read -r ko; do
	mod="${ko#/lib/modules/$kver/kernel/}"; mod="${mod%.ko}"
	objcopy -O binary --only-section=__ksymtab_strings "$ko" /dev/stdout 2>/dev/null |
		tr '\0' '\n' | paste - - |
		awk -F'\t' -v m="$mod" '$1!="" {print "0x00000000\t" $1 "\t" m "\tEXPORT_SYMBOL_GPL\t" $2}'
done >> "$out"
