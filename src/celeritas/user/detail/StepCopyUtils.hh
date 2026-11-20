//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCopyUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class IsSelected>
size_type count_num_valid(IsSelected const& is_selected)
{
    size_type count{0};
    for (TrackSlotId tid : range(TrackSlotId{is_selected.size()}))
    {
        if (is_selected(tid))
        {
            ++count;
        }
    }
    return count;
}

//---------------------------------------------------------------------------//
template<class DstVec, class T, class IsSelected>
void assign_field(
    DstVec* dst,
    StateCollection<T, Ownership::reference, MemSpace::native> const& src,
    IsSelected const& is_selected,
    size_type size)
{
    if (src.empty())
    {
        dst->clear();
        return;
    }

    dst->resize(size);
    auto iter = dst->begin();
    for (TrackSlotId tid : range(TrackSlotId{is_selected.size()}))
    {
        if (is_selected(tid))
        {
            *iter++ = src[tid];
        }
    }
    CELER_ASSERT(iter == dst->end());
}

//---------------------------------------------------------------------------//
template<class DstVec, class T, class IsSelected>
void assign_field(
    DstVec* dst,
    Collection<T, Ownership::reference, MemSpace::native> const& src,
    IsSelected const& is_selected,
    size_type size,
    size_type per_thread)
{
    if (src.empty())
    {
        dst->clear();
        return;
    }

    dst->resize(size * per_thread);
    auto iter = dst->begin();
    for (TrackSlotId tid : range(TrackSlotId{is_selected.size()}))
    {
        if (is_selected(tid))
        {
            for (size_type i = 0; i != per_thread; ++i)
            {
                *iter++ = src[ItemId<T>{per_thread * tid.unchecked_get() + i}];
            }
        }
    }
    CELER_ASSERT(iter == dst->end());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
