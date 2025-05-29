//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
template<Ownership W, MemSpace M>
struct SDParamsData
{
    //// DATA ////

    //! Mapping for volume -> sensitive detector
    Collection<DetectorId, W, M, VolumeId> detector;

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return (!detector.empty());
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SDParamsData& operator=(SDParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        detector = other.detector;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
