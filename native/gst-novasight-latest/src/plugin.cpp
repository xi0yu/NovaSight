#include "gstnslatestgate.hpp"
#include "gstnsinferack.hpp"
#include "ns_frame_token_qdata.hpp"

#include "ns_frame_token.hpp"

namespace {

void destroy_frame_token(gpointer data) {
    delete static_cast<novasight::latest::NsFrameToken*>(data);
}

}  // namespace

GQuark ns_frame_token_qdata_quark() {
    return g_quark_from_static_string("novasight-frame-token");
}

void ns_frame_token_qdata_attach(GstBuffer* buffer, const novasight::latest::NsFrameToken& token) {
    gst_mini_object_set_qdata(
        GST_MINI_OBJECT(buffer),
        ns_frame_token_qdata_quark(),
        new novasight::latest::NsFrameToken(token),
        destroy_frame_token
    );
}

const novasight::latest::NsFrameToken* ns_frame_token_qdata_get(GstBuffer* buffer) {
    return static_cast<const novasight::latest::NsFrameToken*>(
        gst_mini_object_get_qdata(GST_MINI_OBJECT(buffer), ns_frame_token_qdata_quark())
    );
}

static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(plugin, "nslatestgate", GST_RANK_NONE, GST_TYPE_NS_LATEST_GATE) &&
           gst_element_register(plugin, "nsinferack", GST_RANK_NONE, GST_TYPE_NS_INFER_ACK);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    novasightlatest,
    "NovaSight latest-only DeepStream admission gate",
    plugin_init,
    "0.1.0",
    "Proprietary",
    "NovaSight",
    "https://github.com/xi0yu/NovaSight"
)
