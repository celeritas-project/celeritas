//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCopyUtils.device.hh
//---------------------------------------------------------------------------//
#pragma once

#include <thrust/iterator/counting_iterator.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/DeviceAlgorithms.device.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/sys/Stream.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class IdT>
struct IsValidAt
{
    IdT const* ids;

    CELER_FORCEINLINE_FUNCTION bool operator()(size_type i) const
    {
        return static_cast<bool>(ids[i]);
    }
};

//---------------------------------------------------------------------------//
template<class StateData, class IdT>
void select_valid_ids(
    StateData const& state,
    StateCollection<IdT, Ownership::reference, MemSpace::native> const& stencil)
{
    copy_if_async(thrust::make_counting_iterator(size_type{0}),
                  state.valid_id.data(),
                  state.size(),
                  state.num_selected.data(),
                  IsValidAt<IdT>{stencil.data().get()},
                  state.stream_id);
}

//---------------------------------------------------------------------------//
template<class DstVec, class T, class I>
void copy_field(
    DstVec* dst,
    Collection<T, Ownership::reference, MemSpace::native, I> const& src,
    size_type count,
    Stream const& stream)
{
    using DstT = typename DstVec::value_type;
    static_assert(sizeof(DstT) == sizeof(T),
                  "Asynchronous step copies require equal-sized types");
    if (src.empty() || count == 0)
    {
        dst->clear();
        return;
    }
    CELER_EXPECT(count <= src.size());
    dst->resize(count);
    copy_bytes<MemSpace::host, MemSpace::device>(
        ObserverPtr<DstT, MemSpace::host>{dst->data()},
        src.data(),
        count * sizeof(T),
        stream);
}

//---------------------------------------------------------------------------//
template<class DstVec, class SrcRef>
void reserve_field(DstVec* dst, SrcRef const& src)
{
    if (!src.empty())
    {
        dst->reserve(src.size());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
