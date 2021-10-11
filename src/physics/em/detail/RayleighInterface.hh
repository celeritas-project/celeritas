//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RayleighInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Collection.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Rayleigh angular parameters to fit tabulated form factors (\em FF)
 * \f[
 *  FF(E,cos)^2 = \Sigma_{j} \frac{a_j}{[1 + b_j x]^{n}}
 * \f]
 * where \f$ x= E^{2}(1-cos\theta) \f$ and \em n is the high energy slope of
 * the form factor and \em a and \em b are free parameters to obtain the best
 * fit to the form factor. The unit for the energy (\em E) is in MeV.
 */
struct RayleighParameters
{
    Real3 a;
    Real3 b;
    Real3 n;
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
template<Ownership W, MemSpace M>
struct RayleighGroup
{
    //! Model ID
    ModelId model_id;

    //! ID of a gamma
    ParticleId gamma_id;

    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;
    ElementItems<RayleighParameters> params;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && gamma_id && !params.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RayleighGroup& operator=(const RayleighGroup<W2, M2>& other)
    {
        CELER_EXPECT(other);
        model_id = other.model_id;
        gamma_id = other.gamma_id;
        params   = other.params;
        return *this;
    }
};

using RayleighDeviceRef
    = RayleighGroup<Ownership::const_reference, MemSpace::device>;
using RayleighHostRef
    = RayleighGroup<Ownership::const_reference, MemSpace::host>;
using RayleighNativeRef
    = RayleighGroup<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
