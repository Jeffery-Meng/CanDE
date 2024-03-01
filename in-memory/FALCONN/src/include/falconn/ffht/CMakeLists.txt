cmake_minimum_required(VERSION 3.0)

project(ffht)

set(ffht_hdr
        "fast_copy.h"
        "fht.h"
        "fht_impl.h"
        )
set(ffht_src
        "fast_copy.c"
        "fht.c"
        "fht_avx.c"
        "fht_sse.c"
        )

add_library(ffht
        "${ffht_hdr}"
        "${ffht_src}")

set_target_properties(ffht PROPERTIES LINKER_LANGUAGE CXX)





