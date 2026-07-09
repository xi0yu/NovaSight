#pragma once

#include <gst/gst.h>

G_BEGIN_DECLS

#define GST_TYPE_NS_INFER_ACK (gst_ns_infer_ack_get_type())
G_DECLARE_FINAL_TYPE(GstNsInferAck, gst_ns_infer_ack, GST, NS_INFER_ACK, GstElement)

G_END_DECLS
