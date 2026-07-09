#pragma once

#include <cstdint>

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_NS_LATEST_GATE (gst_ns_latest_gate_get_type())
G_DECLARE_FINAL_TYPE(GstNsLatestGate, gst_ns_latest_gate, GST, NS_LATEST_GATE, GstElement)

bool gst_ns_latest_gate_ack(
    GstNsLatestGate* gate,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
);

G_END_DECLS
