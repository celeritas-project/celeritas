//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/DetectorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "geocel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Sensitive detector mapping for geometry
template<Ownership W, MemSpace M>
struct DetectorParamsData
{
    template<class T>
    using ImplVolumeItems = Collection<T, W, M, ImplVolumeId>;

    //// DATA ////

    //! Map implementation volume -> sensitive detector
    ImplVolumeItems<DetectorId> detectors;

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !detectors.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DetectorParamsData& operator=(DetectorParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        detectors = other.detectors;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
