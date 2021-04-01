//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Rayleigh.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/Collection.hh"
#include "physics/base/Types.hh"

namespace celeritas
{
struct ModelInteractPointers;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Rayleigh angular parameters (form factor) for sampling the angular
 * distribution of coherently scattered photon
 */
struct RayleighData
{
    static const unsigned int num_parameters = 9;
    static const unsigned int num_elements   = 100;

    static const real_type angular_parameters[num_parameters][num_elements];
};

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

    //! Rayleigh angular parameters
    using ElementId = celeritas::ItemId<unsigned int>;

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
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Livermore photoelectric interaction
void rayleigh_interact(const RayleighDeviceRef&     pointers,
                       const ModelInteractPointers& model);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
