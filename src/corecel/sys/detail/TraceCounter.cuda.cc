//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/TraceCounter.cuda.cc
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//

#include <cstdint>
#include <type_traits>
#include <nvtx3/nvToolsExt.h>

#include "NvtxUtils.hh"
#include "TraceCounterImpl.hh"

namespace celeritas
{
namespace detail
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
void trace_counter_impl(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>, "Only support numeric counters");

    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = 0xFFFF0000;

    attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = message_handle_for(name);
#define TC_SET_PAYLOAD(TYPE, NVTX_CONST, MEMBER) \
    if constexpr (std::is_same_v<T, TYPE>)       \
    {                                            \
        attributes.payloadType = NVTX_CONST;     \
        attributes.payload.MEMBER = value;       \
    }
    // clang-format off
    TC_SET_PAYLOAD     (std::uint32_t, NVTX_PAYLOAD_TYPE_UNSIGNED_INT32, uiValue)
    else TC_SET_PAYLOAD(std::uint64_t, NVTX_PAYLOAD_TYPE_UNSIGNED_INT64, ullValue)
    else TC_SET_PAYLOAD(float,         NVTX_PAYLOAD_TYPE_FLOAT,          fValue)
    else TC_SET_PAYLOAD(double,        NVTX_PAYLOAD_TYPE_DOUBLE,         dValue)
    else TC_SET_PAYLOAD(std::int32_t,  NVTX_PAYLOAD_TYPE_INT32,          iValue)
    else TC_SET_PAYLOAD(std::int64_t,  NVTX_PAYLOAD_TYPE_INT64,          llValue)
    // clang-format on
#undef TC_SET_PAYLOAD
        nvtxDomainMarkEx(domain_handle(), &attributes);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//

template void trace_counter_impl(char const*, std::uint32_t);
template void trace_counter_impl(char const*, std::uint64_t);
template void trace_counter_impl(char const*, std::int32_t);
template void trace_counter_impl(char const*, std::int64_t);
template void trace_counter_impl(char const*, float);
template void trace_counter_impl(char const*, double);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
