#!/usr/bin/env bash
# Row c: fast-source concentration; flood on both sockets, m allowed reroute senders.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
preflight 1
launch "$ROOT/row_c" 27 --ablation-row c --ablation-floods "0:24,1:24" --ablation-m 8,4,2,1
