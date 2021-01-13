//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SpanRemapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Utility for transforming host to device pointers.
 *
 * Although this is meant for host->device mapping, it will work for any two
 * ranges, so the nomenclature inside the class is "source" and "destination",
 * corresponding to inputs and outputs for the call operator.
 *
 * This class is meant to simplify the construction of device pointer objects,
 * where an array of objects on the device must point to another array of
 * device objects. Rather than using an index and base pointer and copy the
 * index directly to device, you can translate the span from host to device
 * ranges.
 *
 * \code
    auto remap_span = make_span_remapper(make_span(host_vec_),
                                         device_vec_.device_pointers());

    // Create local vector of objects that will be copied to device (and must
    // have device pointers)
    std::vector<Bar> temp_device_obj(host_obj.size());
    for (auto i : range(host_obj.size()))
    {
        temp_device_obj[i].subspan = remap_span(host_obj[i].subspan);
    }
    device_obj.copy_to_device(make_span(temp_device_obj));
   \endcode
 *
 * The extra templating (rather than using T == U == V) simplifies conversions
 * between const and non-const types, as well as (potentially) device pointers
 * and non-device.
 */
template<typename T, typename U>
class SpanRemapper
{
  public:
    //!@{
    //! Type aliases
    using src_type = Span<T>;
    using dst_type = Span<U>;
    //!@}

  public:
    // Construct with source and destination ranges
    inline SpanRemapper(src_type src_span, dst_type dst_span);

    // Convert a subspan of the "source" to a corresponding subspan in "dst"
    template<typename V>
    inline auto operator()(Span<V> src_subspan) const -> dst_type;

  private:
    src_type src_;
    dst_type dst_;
};

//---------------------------------------------------------------------------//
//! Helper function for creating a span mapper.
template<typename T, typename U>
inline SpanRemapper<T, U> make_span_remapper(Span<T> src, Span<U> dst)
{
    return SpanRemapper<T, U>{src, dst};
}

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "SpanRemapper.i.hh"
