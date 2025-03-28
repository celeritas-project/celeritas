//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/TraceCounterImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
void trace_counter_impl(char const* name, T value);

// Explicit instantiations
extern template void trace_counter_impl(char const*, std::uint32_t);
extern template void trace_counter_impl(char const*, std::uint64_t);
extern template void trace_counter_impl(char const*, std::int32_t);
extern template void trace_counter_impl(char const*, std::int64_t);
extern template void trace_counter_impl(char const*, float);
extern template void trace_counter_impl(char const*, double);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas