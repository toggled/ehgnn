#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/third_party"
mkdir -p "$VENDOR"

checkout_repository() {
    local url="$1"
    local destination="$2"
    local commit="$3"

    if [[ ! -d "$destination/.git" ]]; then
        git clone "$url" "$destination"
    fi
    git -C "$destination" fetch --quiet origin "$commit"
    git -C "$destination" checkout --quiet --detach "$commit"
    test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

checkout_repository \
    https://github.com/pkualpha/HSL.git \
    "$VENDOR/HSL" \
    00b181d6c186bf0317958df5e6cb72ecee71f616

checkout_repository \
    https://github.com/youjibiying/HERALD.git \
    "$VENDOR/HERALD" \
    35d0425784ddfec8757e5828d12b1bf6738ffaab

HSL_MODELS="$VENDOR/HSL/models/models.py"
HSL_UPSTREAM_SHA="aeb201d914a254a3258cdc0a1e60feaf3da28d44a06c231305d2a350bb5ce081"
HSL_PATCHED_SHA="3647c054e8b102a5c89e34fb2d1e376afe0c819247c13880cd984ee7fa9f23da"
HSL_OBSERVED_SHA="$(sha256sum "$HSL_MODELS" | awk '{print $1}')"

if [[ "$HSL_OBSERVED_SHA" == "$HSL_UPSTREAM_SHA" ]]; then
    git -C "$VENDOR/HSL" apply \
        "$ROOT/third_party_patches/hsl_forward_mask.patch"
elif [[ "$HSL_OBSERVED_SHA" != "$HSL_PATCHED_SHA" ]]; then
    echo "Unexpected HSL models.py state: $HSL_OBSERVED_SHA" >&2
    exit 1
fi

test "$(sha256sum "$HSL_MODELS" | awk '{print $1}')" = "$HSL_PATCHED_SHA"

echo "Pinned HSL and HERALD sources, including the documented HSL correction, are ready under third_party/."
