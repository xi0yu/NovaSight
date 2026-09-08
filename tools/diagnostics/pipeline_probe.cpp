// Diagnostic-only LD_PRELOAD probe. No image mapping, inference, or device output.
// Build on Jetson; see the P0 report for the exact command and loaded identities.
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvdsinfer_custom_impl.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct Row {
    unsigned stage;
    std::uint64_t start, end, pts, auxiliary;
    long thread;
};
struct Trace {
    std::mutex mutex;
    std::vector<std::string> names;
    std::vector<Row> rows;
    std::uint64_t dropped = 0;
    Trace() { rows.reserve(250000); }
};
Trace& trace() { static auto* value = new Trace; return *value; }
std::uint64_t now_ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return std::uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
unsigned stage(const std::string& name) {
    auto& t = trace();
    std::lock_guard<std::mutex> lock(t.mutex);
    for (unsigned i = 0; i < t.names.size(); ++i)
        if (t.names[i] == name) return i;
    t.names.push_back(name);
    return t.names.size() - 1;
}
void record(unsigned id, std::uint64_t start, std::uint64_t end,
            std::uint64_t pts, std::uint64_t auxiliary) {
    auto& t = trace();
    std::lock_guard<std::mutex> lock(t.mutex);
    // ponytail: bounded diagnostic trace; shorten the run if this cap is reached.
    if (t.rows.size() == 250000) { ++t.dropped; return; }
    t.rows.push_back({id, start, end, pts, auxiliary, syscall(SYS_gettid)});
}
struct Probe { unsigned id; bool meta; };
GstPadProbeReturn probe(GstPad*, GstPadProbeInfo* info, gpointer user) {
    const auto begin = now_ns();
    auto* p = static_cast<Probe*>(user);
    auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) return GST_PAD_PROBE_OK;
    auto pts = GST_BUFFER_PTS(buffer);
    std::uint64_t auxiliary = gst_buffer_get_size(buffer);
    if (p->meta) {
        auto* batch = gst_buffer_get_nvds_batch_meta(buffer);
        if (batch && batch->frame_meta_list) {
            auto* frame = static_cast<NvDsFrameMeta*>(batch->frame_meta_list->data);
            pts = frame->buf_pts;
            auxiliary = frame->num_obj_meta;
        }
    }
    record(p->id, begin, begin, pts, auxiliary);
    return GST_PAD_PROBE_OK;
}
void attach(GstElement* element) {
    if (!GST_IS_BIN(element)) return;
    if (g_object_get_data(G_OBJECT(element), "novasight-p0-probes")) return;
    g_object_set_data(G_OBJECT(element), "novasight-p0-probes", GINT_TO_POINTER(1));
    auto* iterator = gst_bin_iterate_recurse(GST_BIN(element));
    GValue value = G_VALUE_INIT;
    bool done = false;
    while (!done) {
        switch (gst_iterator_next(iterator, &value)) {
        case GST_ITERATOR_OK: {
            auto* child = GST_ELEMENT(g_value_get_object(&value));
            auto* factory = gst_element_get_factory(child);
            if (factory) {
                const std::string type = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
                const std::string name = GST_ELEMENT_NAME(child);
                for (const char* side : {"sink", "src", "sink_0"}) {
                    auto* pad = gst_element_get_static_pad(child, side);
                    if (!pad) continue;
                    auto* p = new Probe{stage(type + "/" + name + "/" + side),
                        type == "nvinfer" && std::string(side) == "src"};
                    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, probe, p,
                        [](gpointer data) { delete static_cast<Probe*>(data); });
                    gst_object_unref(pad);
                }
            }
            g_value_reset(&value);
            break;
        }
        case GST_ITERATOR_RESYNC: gst_iterator_resync(iterator); break;
        default: done = true; break;
        }
    }
    g_value_unset(&value);
    gst_iterator_free(iterator);
}
using Parser = bool (*)(const std::vector<NvDsInferLayerInfo>&,
    const NvDsInferNetworkInfo&, const NvDsInferParseDetectionParams&,
    std::vector<NvDsInferObjectDetectionInfo>&);
Parser original_parser(const char* name) {
    static void* handle = [] {
        const char* path = std::getenv("NOVASIGHT_P0_ORIGINAL_PARSER");
        return path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
    }();
    return handle ? reinterpret_cast<Parser>(dlsym(handle, name)) : nullptr;
}
}

extern "C" GstElement* gst_parse_launch(const gchar* description, GError** error) {
    using Launch = GstElement* (*)(const gchar*, GError**);
    static auto launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "gst_parse_launch"));
    if (!launch) return nullptr;
    auto* element = launch(description, error);
    if (element) attach(element);
    return element;
}

extern "C" GstElement* gst_parse_launchv(const gchar** argv, GError** error) {
    using Launch = GstElement* (*)(const gchar**, GError**);
    static auto launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "gst_parse_launchv"));
    if (!launch) return nullptr;
    auto* element = launch(argv, error);
    if (element) attach(element);
    return element;
}

#define WRAP_PARSER(NAME) \
extern "C" bool NAME(const std::vector<NvDsInferLayerInfo>& layers, \
    const NvDsInferNetworkInfo& network, const NvDsInferParseDetectionParams& params, \
    std::vector<NvDsInferObjectDetectionInfo>& objects) { \
    static auto original = original_parser(#NAME); \
    static auto id = stage("cpu_parser/" #NAME); \
    if (!original) { std::fprintf(stderr, "P0_ORIGINAL_PARSER_UNAVAILABLE\n"); return false; } \
    const auto start = now_ns(); \
    bool ok = original(layers, network, params, objects); \
    const auto end = now_ns(); \
    record(id, start, end, GST_CLOCK_TIME_NONE, objects.size()); \
    return ok; \
}
WRAP_PARSER(NvDsInferParseNovaSightRaw)
WRAP_PARSER(NvDsInferParseNovaSightDecodedNms)
WRAP_PARSER(NvDsInferParseNovaSightEfficientNms)
WRAP_PARSER(NvDsInferParseNovaSightRockchipYoloV5)

// Optional scopes from isolated diagnostic builds. No dependency in production.
extern "C" void novasight_p0_span(const char* name, std::uint64_t start,
    std::uint64_t end, std::uint64_t identity, std::uint64_t auxiliary) {
    record(stage(name), start, end, identity, auxiliary);
}

extern "C" void novasight_p0_flush() {
    const char* path = std::getenv("NOVASIGHT_P0_TRACE");
    if (!path) return;
    auto& t = trace();
    std::lock_guard<std::mutex> lock(t.mutex);
    auto* file = std::fopen(path, "w");
    if (!file) { std::perror("P0_TRACE_OPEN"); return; }
    std::fprintf(file, "# dropped=%llu\n", (unsigned long long)t.dropped);
    std::fputs("stage,start_ns,end_ns,pts,auxiliary,thread\n", file);
    for (const auto& row : t.rows)
        std::fprintf(file, "%s,%llu,%llu,%llu,%llu,%ld\n", t.names[row.stage].c_str(),
            (unsigned long long)row.start, (unsigned long long)row.end,
            (unsigned long long)row.pts, (unsigned long long)row.auxiliary, row.thread);
    if (std::fclose(file)) std::perror("P0_TRACE_CLOSE");
}
__attribute__((destructor)) static void flush_trace() { novasight_p0_flush(); }
