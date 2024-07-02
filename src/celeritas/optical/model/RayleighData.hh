//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighData.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
template <Ownership W, MemSpace M>
struct RayleighData
{
    template <class T>
    using Items = Collection<T, W, M>;
    template <class T>
    using OpticalMaterialItems = Collection<T, W, M, OpticalMaterialId>;


    OpticalMaterialItems<real_type> scale_factor;
    OpticalMaterialItems<real_type> compressibility;


    Items<real_type> reals;



    explicit CELER_FUNCTION operator bool() const
    {
    }

    template <Ownership W2, MemSpace M2>
    RayleighData& operator=(RayleighData<W2,M2> const& other)
    {
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
