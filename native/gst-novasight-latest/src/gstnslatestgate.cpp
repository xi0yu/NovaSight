#include "gstnslatestgate.hpp"

#include "ns_frame_token.hpp"
#include "ns_frame_token_qdata.hpp"
#include "ns_gate_registry.hpp"
#include "ns_latest_gate_core.hpp"

#include <cstring>

using novasight::latest::NsAckResult;
using novasight::latest::NsEosPolicy;
using novasight::latest::NsGateConfig;
using novasight::latest::NsGateError;
using novasight::latest::NsGateInput;
using novasight::latest::NsLatestGateCore;

struct _GstNsLatestGate {
    GstElement parent;

    GstPad* sinkpad;
    GstPad* srcpad;

    GMutex lock;
    GCond cond;
    GThread* dispatch_thread;

    gchar* gate_id;
    GstBuffer* pending_buffer;
    NsLatestGateCore* core;

    gboolean dispatch_running;
    gboolean stopping;
};

G_DEFINE_TYPE(GstNsLatestGate, gst_ns_latest_gate, GST_TYPE_ELEMENT)

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

static bool latest_gate_ack_callback(
    void* owner,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
) {
    auto* gate = GST_NS_LATEST_GATE(owner);
    return gst_ns_latest_gate_ack(gate, session_epoch, generation, ack_ns);
}

static void clear_pending_locked(GstNsLatestGate* gate) {
    if (gate->pending_buffer != nullptr) {
        gst_buffer_unref(gate->pending_buffer);
        gate->pending_buffer = nullptr;
    }
}

static gpointer dispatch_loop(gpointer data) {
    auto* gate = GST_NS_LATEST_GATE(data);

    while (true) {
        GstBuffer* selected = nullptr;

        g_mutex_lock(&gate->lock);
        while (!gate->stopping) {
            gate->core->check_ack_timeout(monotonic_now_ns());
            auto snapshot = gate->core->snapshot();
            if (snapshot.error) {
                GST_ELEMENT_ERROR(
                    gate,
                    STREAM,
                    FAILED,
                    ("NovaSight latest gate entered error state"),
                    ("%s", gate->core->error_string().c_str())
                );
                gate->stopping = TRUE;
                break;
            }
            if (gate->pending_buffer != nullptr &&
                snapshot.inference_credit &&
                !snapshot.inference_inflight &&
                !snapshot.flushing) {
                selected = gate->pending_buffer;
                gate->pending_buffer = nullptr;
                auto token = gate->core->dispatch_next(monotonic_now_ns());
                if (!token.has_value()) {
                    gst_buffer_unref(selected);
                    selected = nullptr;
                    continue;
                }
                ns_frame_token_qdata_attach(selected, *token);
                break;
            }
            g_cond_wait_until(&gate->cond, &gate->lock, g_get_monotonic_time() + 10'000);
        }
        const gboolean stopping = gate->stopping;
        g_mutex_unlock(&gate->lock);

        if (stopping) {
            if (selected != nullptr) {
                gst_buffer_unref(selected);
            }
            break;
        }
        if (selected == nullptr) {
            continue;
        }

        GstFlowReturn result = gst_pad_push(gate->srcpad, selected);
        if (result != GST_FLOW_OK) {
            g_mutex_lock(&gate->lock);
            gate->core->mark_push_failed();
            gate->stopping = TRUE;
            g_cond_broadcast(&gate->cond);
            g_mutex_unlock(&gate->lock);
        }
    }

    return nullptr;
}

static GstFlowReturn gst_ns_latest_gate_chain(GstPad* pad, GstObject* parent, GstBuffer* buffer) {
    (void)pad;
    auto* gate = GST_NS_LATEST_GATE(parent);
    GstFlowReturn result = GST_FLOW_OK;

    g_mutex_lock(&gate->lock);
    auto snapshot = gate->core->snapshot();
    if (gate->stopping || snapshot.flushing) {
        result = GST_FLOW_FLUSHING;
    } else if (snapshot.error) {
        result = GST_FLOW_ERROR;
    } else {
        GstClockTime pts = GST_BUFFER_PTS(buffer);
        gate->core->submit(NsGateInput{
            snapshot.received_generation + 1,
            GST_CLOCK_TIME_IS_VALID(pts) ? static_cast<std::uint64_t>(pts) : 0ULL,
            monotonic_now_ns(),
        });
        clear_pending_locked(gate);
        gate->pending_buffer = buffer;
        buffer = nullptr;
        g_cond_broadcast(&gate->cond);
    }
    g_mutex_unlock(&gate->lock);

    if (buffer != nullptr) {
        gst_buffer_unref(buffer);
    }
    return result;
}

static gboolean gst_ns_latest_gate_event(GstPad* pad, GstObject* parent, GstEvent* event) {
    (void)pad;
    auto* gate = GST_NS_LATEST_GATE(parent);

    switch (GST_EVENT_TYPE(event)) {
        case GST_EVENT_FLUSH_START:
            g_mutex_lock(&gate->lock);
            gate->core->flush_start();
            clear_pending_locked(gate);
            g_cond_broadcast(&gate->cond);
            g_mutex_unlock(&gate->lock);
            return gst_pad_push_event(gate->srcpad, event);
        case GST_EVENT_FLUSH_STOP:
            g_mutex_lock(&gate->lock);
            gate->core->flush_stop();
            gate->stopping = FALSE;
            g_cond_broadcast(&gate->cond);
            g_mutex_unlock(&gate->lock);
            return gst_pad_push_event(gate->srcpad, event);
        case GST_EVENT_EOS:
            g_mutex_lock(&gate->lock);
            gate->core->eos();
            clear_pending_locked(gate);
            g_cond_broadcast(&gate->cond);
            g_mutex_unlock(&gate->lock);
            return gst_pad_push_event(gate->srcpad, event);
        default:
            return gst_pad_push_event(gate->srcpad, event);
    }
}

bool gst_ns_latest_gate_ack(
    GstNsLatestGate* gate,
    std::uint64_t session_epoch,
    std::uint64_t generation,
    std::uint64_t ack_ns
) {
    g_return_val_if_fail(GST_IS_NS_LATEST_GATE(gate), false);
    g_mutex_lock(&gate->lock);
    NsAckResult result = gate->core->ack(session_epoch, generation, ack_ns);
    if (result == NsAckResult::Accepted) {
        g_cond_broadcast(&gate->cond);
    }
    bool accepted = result == NsAckResult::Accepted;
    g_mutex_unlock(&gate->lock);
    return accepted;
}

static GstStateChangeReturn gst_ns_latest_gate_change_state(
    GstElement* element,
    GstStateChange transition
) {
    auto* gate = GST_NS_LATEST_GATE(element);

    if (transition == GST_STATE_CHANGE_READY_TO_PAUSED) {
        g_mutex_lock(&gate->lock);
        gate->stopping = FALSE;
        gate->core->reset();
        gate->dispatch_running = TRUE;
        ns_gate_registry_register(gate->gate_id, gate, latest_gate_ack_callback);
        gate->dispatch_thread = g_thread_new("nslatestgate-dispatch", dispatch_loop, gate);
        g_mutex_unlock(&gate->lock);
    }

    GstStateChangeReturn ret = GST_ELEMENT_CLASS(gst_ns_latest_gate_parent_class)
        ->change_state(element, transition);

    if (transition == GST_STATE_CHANGE_PAUSED_TO_READY) {
        g_mutex_lock(&gate->lock);
        gate->stopping = TRUE;
        gate->core->stop();
        clear_pending_locked(gate);
        g_cond_broadcast(&gate->cond);
        GThread* thread = gate->dispatch_thread;
        gate->dispatch_thread = nullptr;
        gate->dispatch_running = FALSE;
        ns_gate_registry_unregister(gate->gate_id, gate);
        g_mutex_unlock(&gate->lock);
        if (thread != nullptr) {
            g_thread_join(thread);
        }
    }

    return ret;
}

static void gst_ns_latest_gate_set_property(
    GObject* object,
    guint prop_id,
    const GValue* value,
    GParamSpec* pspec
) {
    auto* gate = GST_NS_LATEST_GATE(object);
    switch (prop_id) {
        case PROP_GATE_ID:
            g_free(gate->gate_id);
            gate->gate_id = g_value_dup_string(value);
            if (gate->gate_id == nullptr || gate->gate_id[0] == '\0') {
                g_free(gate->gate_id);
                gate->gate_id = g_strdup("default");
            }
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_ns_latest_gate_get_property(
    GObject* object,
    guint prop_id,
    GValue* value,
    GParamSpec* pspec
) {
    auto* gate = GST_NS_LATEST_GATE(object);
    switch (prop_id) {
        case PROP_GATE_ID:
            g_value_set_string(value, gate->gate_id);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void gst_ns_latest_gate_finalize(GObject* object) {
    auto* gate = GST_NS_LATEST_GATE(object);
    g_mutex_lock(&gate->lock);
    gate->stopping = TRUE;
    clear_pending_locked(gate);
    g_cond_broadcast(&gate->cond);
    GThread* thread = gate->dispatch_thread;
    gate->dispatch_thread = nullptr;
    ns_gate_registry_unregister(gate->gate_id, gate);
    g_mutex_unlock(&gate->lock);
    if (thread != nullptr) {
        g_thread_join(thread);
    }
    delete gate->core;
    g_free(gate->gate_id);
    g_cond_clear(&gate->cond);
    g_mutex_clear(&gate->lock);
    G_OBJECT_CLASS(gst_ns_latest_gate_parent_class)->finalize(object);
}

static void gst_ns_latest_gate_class_init(GstNsLatestGateClass* klass) {
    auto* object_class = G_OBJECT_CLASS(klass);
    auto* element_class = GST_ELEMENT_CLASS(klass);

    object_class->set_property = gst_ns_latest_gate_set_property;
    object_class->get_property = gst_ns_latest_gate_get_property;
    object_class->finalize = gst_ns_latest_gate_finalize;
    element_class->change_state = gst_ns_latest_gate_change_state;

    g_object_class_install_property(
        object_class,
        PROP_GATE_ID,
        g_param_spec_string(
            "gate-id",
            "Gate ID",
            "Shared gate identifier used by nsinferack to release inference credit",
            "default",
            static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
        )
    );

    gst_element_class_set_static_metadata(
        element_class,
        "NovaSight Latest Admission Gate",
        "Filter/Video",
        "Single-credit latest-only admission gate for DeepStream inference",
        "NovaSight"
    );
    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);
}

static void gst_ns_latest_gate_init(GstNsLatestGate* gate) {
    gate->sinkpad = gst_pad_new_from_static_template(&sink_template, "sink");
    gate->srcpad = gst_pad_new_from_static_template(&src_template, "src");
    gst_pad_set_chain_function(gate->sinkpad, GST_DEBUG_FUNCPTR(gst_ns_latest_gate_chain));
    gst_pad_set_event_function(gate->sinkpad, GST_DEBUG_FUNCPTR(gst_ns_latest_gate_event));
    gst_element_add_pad(GST_ELEMENT(gate), gate->sinkpad);
    gst_element_add_pad(GST_ELEMENT(gate), gate->srcpad);

    g_mutex_init(&gate->lock);
    g_cond_init(&gate->cond);
    gate->gate_id = g_strdup("default");
    gate->pending_buffer = nullptr;
    gate->dispatch_thread = nullptr;
    gate->dispatch_running = FALSE;
    gate->stopping = FALSE;
    gate->core = new NsLatestGateCore(NsGateConfig{200'000'000ULL, NsEosPolicy::DropPending});
}
