//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TraceCounter.cuda.cc
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//
#include "TraceCounter.hh"

#include <cstdint>
#include <type_traits>
#include <nvtx3/nvToolsExt.h>

#include "corecel/sys/detail/NvtxUtils.hh"

#include "ScopedProfiling.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Simple performance tracing counter.
 * \tparam T Arithmetic counter type
 *
 * Records a named value at the current timestamp which
 * can then be displayed on a timeline. Only supported on host
 */
template<class T>
void trace_counter(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");
    if (use_profiling())
    {
        nvtxEventAttributes_t attributes{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = 0xFFFF0000;

        attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        attributes.message.registered = message_handle_for(name);
        if constexpr (std::is_same_v<T, std::uint32_t>)
        {
            attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT32;
            attributes.payload.uiValue = value;
        }
        else if constexpr (std::is_same_v<T, std::uint64_t>)
        {
            attributes.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            attributes.payload.ullValue = static_cast<uint64_t>(value);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            attributes.payloadType = NVTX_PAYLOAD_TYPE_FLOAT;
            attributes.payload.fValue = value;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            attributes.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;
            attributes.payload.dValue = value;
        }
        else if constexpr (std::is_same_v<T, std::int32_t>)
        {
            attributes.payloadType = NVTX_PAYLOAD_TYPE_INT32;
            attributes.payload.iValue = value;
        }
        else if constexpr (std::is_same_v<T, std::int64_t>)
        {
            attributes.payloadType = NVTX_PAYLOAD_TYPE_INT64;
            attributes.payload.llValue = value;
        }
        nvtxDomainMarkEx(domain_handle(), &attributes);
    }
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//

template void trace_counter(char const*, std::uint32_t);
template void trace_counter(char const*, std::uint64_t);
template void trace_counter(char const*, std::int32_t);
template void trace_counter(char const*, std::int64_t);
template void trace_counter(char const*, float);
template void trace_counter(char const*, double);

//---------------------------------------------------------------------------//
}  // namespace celeritas
