//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/data/RayleighData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Rayleigh angular parameters to fit tabulated form factors.
 *
 * The form factors \em FF (constructed by the RayleighModel) are:
 * \f[
 *  FF(E,cos)^2 = \Sigma_{j} \frac{a_j}{[1 + b_j x]^{n}}
 * \f]
 * where \f$ x = E^{2}(1 - \cos\theta) \f$ and \em n is the high energy slope
 * of the form factor and \em a and \em b are free parameters to obtain the
 * best fit to the form factor. The unit for the energy (\em E) is in MeV.
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
struct RayleighData
{
    template<class T>
    using ElementItems = celeritas::Collection<T, W, M, ElementId>;

    ParticleId gamma;
    ElementItems<RayleighParameters> params;

    //! Check whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return gamma && !params.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RayleighData& operator=(RayleighData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        gamma = other.gamma;
        params = other.params;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
