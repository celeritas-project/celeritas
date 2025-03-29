//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/TraceCounterImpl.cuda.cc
//! \brief Numeric tracing counter
//---------------------------------------------------------------------------//

#include "TraceCounterImpl.hh"

#include <cstdint>
#include <type_traits>
#include <nvtx3/nvToolsExt.h>

#include "NvtxUtils.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// Non-templated code for constructing attributes
nvtxEventAttributes_t make_base_attributes(char const* name)
{
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

    attributes.colorType = NVTX_COLOR_ARGB;
    attributes.color = 0xFFFF0000;

    attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attributes.message.registered = message_handle_for(name);
    return attributes;
}

}  // namespace

//---------------------------------------------------------------------------//
//! Implement NVTX counters for CUDA
template<class T>
void trace_counter_impl(char const* name, T value)
{
    static_assert(std::is_arithmetic_v<T>,
                  "Only numeric counters are supported");

    auto attributes = make_base_attributes(name);

#define TC_SET_PAYLOAD(TYPE, NVTX_ENUM, MEMBER)                 \
    if constexpr (std::is_same_v<T, TYPE>)                      \
    {                                                           \
        attributes.payloadType = NVTX_PAYLOAD_TYPE_##NVTX_ENUM; \
        attributes.payload.MEMBER = value;                      \
    }
    // clang-format off
    TC_SET_PAYLOAD(std::uint64_t,      UNSIGNED_INT64, ullValue)
    else TC_SET_PAYLOAD(std::int64_t,  INT64,          llValue)
    else TC_SET_PAYLOAD(double,        DOUBLE,         dValue)
    else TC_SET_PAYLOAD(std::uint32_t, UNSIGNED_INT32, uiValue)
    else TC_SET_PAYLOAD(std::int32_t,  INT32,          iValue)
    else TC_SET_PAYLOAD(float,         FLOAT,          fValue)
    // clang-format on
#undef TC_SET_PAYLOAD

        // Save attributes for this domain
        nvtxDomainMarkEx(domain_handle(), &attributes);
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATIONS
//---------------------------------------------------------------------------//

template void trace_counter_impl(char const*, double);
template void trace_counter_impl(char const*, float);
template void trace_counter_impl(char const*, std::int32_t);
template void trace_counter_impl(char const*, std::int64_t);
template void trace_counter_impl(char const*, std::uint32_t);
template void trace_counter_impl(char const*, std::uint64_t);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
