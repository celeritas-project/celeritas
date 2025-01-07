//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedMem.cc
//---------------------------------------------------------------------------//
#include "ScopedMem.hh"

#if defined(__APPLE__)
#    include <cstring>
#    include <mach/mach.h>
#elif defined(__linux__)
#    include <sys/resource.h>
#elif defined(_WIN32)
#    include <windows.h>
// Note: windows header *must* precede psapi
#    include <psapi.h>
#endif
#include <cstddef>
#include <iostream>

#include "corecel/DeviceRuntimeApi.hh"

#include "corecel/Assert.hh"
#include "corecel/math/Quantity.hh"

#include "Device.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
struct MemResult
{
    std::size_t hwm{0};
    std::size_t resident{0};  // unused, not available on linux
};

//---------------------------------------------------------------------------//
//! Return high water mark and possibly resident memory [bytes]
MemResult get_cpu_mem()
{
    MemResult result;
#if defined(__APPLE__)
    struct mach_task_basic_info tinfo;
    mach_msg_type_number_t tcount = MACH_TASK_BASIC_INFO_COUNT;
    tinfo.resident_size = 0;
    tinfo.resident_size_max = 0;

    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&tinfo),
                  &tcount)
        == KERN_SUCCESS)
    {
        // Units are B
        result.hwm = tinfo.resident_size_max;
        result.resident = tinfo.resident_size;
    }
#elif defined(__linux__)
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    struct rusage usage;
    usage.ru_maxrss = 0;
    if (!getrusage(RUSAGE_SELF, &usage))
    {
        // Units are kiB!
        result.hwm = usage.ru_maxrss * 1024u;
    }
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
#elif defined(_WIN32)
    // Units are B
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    result.hwm = info.PeakWorkingSetSize;
    result.resident = info.WorkingSetSize;
#endif
    return result;
}

std::size_t get_gpu_mem()
{
    std::size_t free{0};
    std::size_t total{0};
    CELER_DEVICE_CALL_PREFIX(MemGetInfo(&free, &total));
    CELER_ASSERT(total > free);
    return total - free;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with name and a pointer to the mem registry.
 */
ScopedMem::ScopedMem(std::string_view label, MemRegistry* registry)
    : registry_(registry)
{
    CELER_EXPECT(registry_.value());
    CELER_EXPECT(!label.empty());

    id_ = registry_.value()->push();
    CELER_ASSERT(id_);

    MemUsageEntry& entry = registry_.value()->get(id_);
    entry.label = label;

    cpu_start_hwm_ = get_cpu_mem().hwm;
    if (celeritas::device())
    {
        gpu_start_used_ = get_gpu_mem();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Register data on destruction.
 */
ScopedMem::~ScopedMem() noexcept(!CELERITAS_DEBUG)
{
    if (registry_.value() != nullptr)
    {
        MemUsageEntry& entry = registry_.value()->get(id_);

        // Save CPU stats
        auto stop_hwm = get_cpu_mem().hwm;
        entry.cpu_hwm = native_value_to<KibiBytes>(stop_hwm);
        if (CELER_UNLIKELY(stop_hwm < cpu_start_hwm_))
        {
            std::cerr << "An error occurred while calculating CPU memory "
                         "usage for '"
                      << entry.label << ": end HWM "
                      << native_value_to<KibiBytes>(stop_hwm).value()
                      << " KiB is less than beginning HWM "
                      << native_value_to<KibiBytes>(cpu_start_hwm_).value()
                      << " KiB\n";
        }
        entry.cpu_delta = native_value_to<KibiBytes>(stop_hwm - cpu_start_hwm_);

        if (celeritas::device())
        {
            std::size_t stop_usage = -1;
            try
            {
                stop_usage = get_gpu_mem();
                entry.gpu_usage = native_value_to<KibiBytes>(stop_usage);
                entry.gpu_delta
                    = native_value_to<KibiBytes>(stop_usage - gpu_start_used_);
            }
            catch (std::exception const& e)
            {
                std::cerr << "An error occurred while calculating GPU memory "
                             "usage for '"
                          << entry.label << " (start = " << gpu_start_used_
                          << ", stop = " << stop_usage << "): " << e.what()
                          << std::endl;
            }
        }

        registry_.value()->pop();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
