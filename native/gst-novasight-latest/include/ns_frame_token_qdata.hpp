#pragma once

#include <gst/gst.h>

#include "ns_frame_token.hpp"

GQuark ns_frame_token_qdata_quark();
void ns_frame_token_qdata_attach(
    GstBuffer* buffer,
    const novasight::latest::NsFrameToken& token
);
const novasight::latest::NsFrameToken* ns_frame_token_qdata_get(GstBuffer* buffer);
