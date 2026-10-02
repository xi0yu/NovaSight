#include "novasight_deepstream_bridge.h"

#include <cstdint>

int main() {
    if (ns_ds_bridge_abi_version() != NS_DS_BRIDGE_ABI_VERSION) {
        return 1;
    }
    if (ns_ds_bridge_frame_snapshot_size() != sizeof(NsDsFrameSnapshot)) {
        return 2;
    }
    if (ns_ds_bridge_status_string(NS_DS_BRIDGE_OK) == nullptr) {
        return 3;
    }
    return 0;
}
