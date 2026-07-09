#include "gstnsinferack.hpp"

#include "ns_frame_token.hpp"
#include "ns_frame_token_qdata.hpp"
#include "ns_gate_registry.hpp"

struct _GstNsInferAck {
    GstElement parent;

    GstPad* sinkpad;
    GstPad* srcpad;
    gchar* gate_id;
};

G_DEFINE_TYPE(GstNsInferAck, gst_ns_infer_ack, GST_TYPE_ELEMENT)

enum {
    PROP_0,
    PROP_GATE_ID,
};

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY
);

static std::uint64_t monotonic_now_ns() {
    return static_cast<std::uint64_t>(g_get_monotonic_time()) * 1000ULL;
}

static GstFlowReturn gst_ns_infer_ack_chain(GstPad* pad, GstObject* parent, GstBuffer* buffer) {
    (void)pad;
    auto* ack = GST_NS_INFER_ACK(parent);
    const auto* token = ns_frame_token_qdata_get(buffer);
    if (token == nullptr) {
        GST_ELEMENT_ERROR(
            ack,
            STREAM,
            FAILED,
            ("NovaSight inference ACK missing frame token"),
            ("nslatestgate did not attach a token or upstream copied the buffer without metadata")
        );
        gst_buffer_unref(buffer);
        return GST_FLOW_ERROR;
    }

    if (!ns_gate_registry_ack(ack->gate_id, token->session_epoch, token->generation, monotonic_now_ns())) {
        GST_ELEMENT_ERROR(
            ack,
            STREAM,
            FAILED,
            ("NovaSight inference ACK could not release latest gate credit"),
            ("gate-id=%s epoch=%" G_GUINT64_FORMAT " generation=%" G_GUINT64_FORMAT,
             ack->gate_id,
             static_cast<guint64>(token->session_epoch),
             static_cast<guint64>(token->generation))
        );
        gst_buffer_unref(buffer);
        return GST_FLOW_ERROR;
    }

    return gst_pad_push(ack->srcpad, buffer);
}

static gboolean gst_ns_infer_ack_event(GstPad* pad, GstObject* parent, GstEvent* event) {
    (void)pad;
    auto* ack = GST_NS_INFER_ACK(parent);
    return gst_pad_push_event(ack->srcpad, event);
}

static void gst_ns_infer_ack_set_property(
    GObject* object,
    guint prop_id,
    const GValue* value,
    GParamSpec* pspec
) {
    auto* ack = GST_NS_INFER_ACK(object);
    switch (prop_id) {
        case PROP_GATE_ID:
            g_free(ack->gate_id);
            ack->gate_id = g_value_dup_string(value);
            if (ack->gate_id == nullptr || ack->gate_id[0] == '\0') {
                g_free(ack->gate_id);
                ack->gate_id = g_strdup("default");
            }
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_ns_infer_ack_get_property(
    GObject* object,
    guint prop_id,
    GValue* value,
    GParamSpec* pspec
) {
    auto* ack = GST_NS_INFER_ACK(object);
    switch (prop_id) {
        case PROP_GATE_ID:
            g_value_set_string(value, ack->gate_id);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_ns_infer_ack_finalize(GObject* object) {
    auto* ack = GST_NS_INFER_ACK(object);
    g_free(ack->gate_id);
    G_OBJECT_CLASS(gst_ns_infer_ack_parent_class)->finalize(object);
}

static void gst_ns_infer_ack_class_init(GstNsInferAckClass* klass) {
    auto* object_class = G_OBJECT_CLASS(klass);
    auto* element_class = GST_ELEMENT_CLASS(klass);

    object_class->set_property = gst_ns_infer_ack_set_property;
    object_class->get_property = gst_ns_infer_ack_get_property;
    object_class->finalize = gst_ns_infer_ack_finalize;

    g_object_class_install_property(
        object_class,
        PROP_GATE_ID,
        g_param_spec_string(
            "gate-id",
            "Gate ID",
            "Shared latest gate identifier whose credit should be released",
            "default",
            static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
        )
    );

    gst_element_class_set_static_metadata(
        element_class,
        "NovaSight Inference ACK",
        "Filter/Video",
        "Releases NovaSight latest gate credit after nvinfer output",
        "NovaSight"
    );
    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);
}

static void gst_ns_infer_ack_init(GstNsInferAck* ack) {
    ack->sinkpad = gst_pad_new_from_static_template(&sink_template, "sink");
    ack->srcpad = gst_pad_new_from_static_template(&src_template, "src");
    gst_pad_set_chain_function(ack->sinkpad, GST_DEBUG_FUNCPTR(gst_ns_infer_ack_chain));
    gst_pad_set_event_function(ack->sinkpad, GST_DEBUG_FUNCPTR(gst_ns_infer_ack_event));
    gst_element_add_pad(GST_ELEMENT(ack), ack->sinkpad);
    gst_element_add_pad(GST_ELEMENT(ack), ack->srcpad);
    ack->gate_id = g_strdup("default");
}
