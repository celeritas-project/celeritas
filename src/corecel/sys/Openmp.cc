//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Openmp.cc
//---------------------------------------------------------------------------//
#include "Openmp.hh"

#include "corecel/Assert.hh"

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//

/*!
 * Get the maximum number of OpenMP threads that could ever execute in
 * parallel.
 *
 * This may be limited by \c OMP_THREAD_LIMIT but also may be uselessly large
 * (i.e., \c INT_MAX).
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu123.html .
 */
OpenmpSize_t openmp_thread_limit()
{
#ifdef _OPENMP
    return omp_get_thread_limit();
#else
    return 1;
#endif
}

/*!
 * Get the maximum number of OpenMP threads that can execute in parallel.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu112.html .
 */
OpenmpSize_t openmp_max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/*!
 * Get the number of CPU threads in the \em current parallel region.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu111.html .
 */
OpenmpSize_t openmp_num_threads()
{
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

/*!
 * Set the default number of threads for default future parallel regions.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu110.html .
 *
 * \note This is named in sync with the OpenMP spec, but it acts more like "max
 * threads" (which is used outside a parallel region).
 */
void openmp_num_threads(OpenmpSize_t num_threads)
{
    CELER_EXPECT(num_threads > 0);
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    CELER_VALIDATE(
        num_threads == 1,
        << R"(cannot set CPU thread limit above 1 when OpenMP is disabled)");
#endif
}

/*!
 * Get the OpenMP thread ID inside a parallel region.
 *
 * If running in a serial application or region, this returns zero.
 * See https://www.openmp.org/spec-html/5.0/openmpsu113.html .
 */
OpenmpThreadId openmp_local_thread()
{
#ifdef _OPENMP
    return id_cast<OpenmpThreadId>(omp_get_thread_num());
#else
    return StreamId{0};
#endif
}

/*!
 * Get the openmp process bind affinity for parallel regions.
 *
 * See https://www.openmp.org/spec-html/5.0/openmpsu132.html .
 */
char const* openmp_proc_bind()
{
#ifdef _OPENMP
    switch (omp_get_proc_bind())
    {
        case omp_proc_bind_false:
            return "false";
        case omp_proc_bind_true:
            return "true";
        case omp_proc_bind_master:
            return "master";
        case omp_proc_bind_close:
            return "close";
        case omp_proc_bind_spread:
            return "spread";
        default:
            CELER_ASSERT_UNREACHABLE();
    }
#else
    return "disabled";
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
