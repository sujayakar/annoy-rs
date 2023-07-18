#include "annoy/src/annoylib.h"
#include "annoy/src/kissrandom.h"

using namespace Annoy;

typedef ::AnnoyIndex<int32_t, float, ::Angular, ::Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy> AngularIndex;

extern "C"
{
    void *annoy_angular_create_index(int f);
    bool annoy_angular_add_item(void *idx, int item, float *w, char **error);
    bool annoy_angular_build(void *idx, int q, int n_threads, char **error);
    bool annoy_angular_unbuild(void *idx, char **error);
    bool annoy_angular_save(void *idx, char *filename, bool prefault, char **error);
    void annoy_angular_unload(void *idx);
    bool annoy_angular_load(void *idx, char *filename, bool prefault, char **error);
    float annoy_angular_get_distance(void *idx, uint32_t i, uint32_t j);
    size_t annoy_angular_get_nns_by_item(void *idx, uint32_t item, size_t n, int search_k, uint32_t *result, float *distances);
    size_t annoy_angular_get_nns_by_vector(void *idx, float *w, size_t n, int search_k, uint32_t *result, float *distances);
    uint32_t annoy_angular_get_n_items(void *idx);
    void annoy_angular_verbose(void *idx, bool v);
    void annoy_angular_get_item(void *idx, uint32_t item, float *v);
    void annoy_angular_set_seed(void *idx, uint64_t seed);
    bool annoy_angular_on_disk_build(void *idx, char *filename, char **error);
    void annoy_angular_free_index(void *idx);
    void annoy_angular_free_error(char *error);
}
