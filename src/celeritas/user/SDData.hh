//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SDData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{

namespace detail
{

template<Ownership W, MemSpace M>
struct SDParamsData
{
    //// DATA ////

    // ! Mapping for volume -> sensitive detector
    Collection<DetectorId, W, M, VolumeId> detector;

    // ! boolean for assignment of struct.
    // ! TODO: this is a placeholder for now since COllection doesn't case to
    // bool and is the only member
    bool assigned;

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(assigned);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SDParamsData& operator=(SDParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        detector = other.detector;
        assigned = other.assigned;
        return *this;
    }
};

}  // namespace detail
}  // namespace celeritas