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
constexpr unsigned int rayleigh_num_parameters = 9;
constexpr unsigned int rayleigh_num_elements   = 100;

struct RayleighData 
{
    static real_type 
        angular_parameters[rayleigh_num_parameters][rayleigh_num_elements]; 
};

//---------------------------------------------------------------------------//
/*!
 * Storage for Rayleigh angular parameters
 */
template<Ownership W, MemSpace M>
struct RayleighParameters
{
    using IntId = celeritas::ItemId<int>;

    template<class T>
    using Items = celeritas::Collection<T, W, M, IntId>;

    Items<Array<real_type, rayleigh_num_parameters>> data;

    explicit CELER_FUNCTION operator bool() const { return !data.empty(); }

    //! Assign from another set of parameters
    template<Ownership W2, MemSpace M2>
    RayleighParameters& operator=(const RayleighParameters<W2, M2>& other)
    {
        CELER_EXPECT(other);
        data = other.data;

        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Device data for creating an interactor.
 */
template<Ownership W, MemSpace M>
struct RayleighPointers
{
    //! Model ID
    ModelId model_id;

    //! ID of a gamma
    ParticleId gamma_id;

    //! Rayleigh angular parameters
    RayleighParameters<W, M> params;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return model_id && gamma_id && params;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RayleighPointers& operator=(const RayleighPointers<W2, M2>& other)
    {
        CELER_EXPECT(other);
        model_id = other.model_id;
        gamma_id = other.gamma_id;
        params   = other.params;
        return *this;
    }
};

using RayleighDeviceRef
    = RayleighPointers<Ownership::const_reference, MemSpace::device>;
using RayleighHostRef
    = RayleighPointers<Ownership::const_reference, MemSpace::host>;
using RayleighNativePointers
    = RayleighPointers<Ownership::const_reference, MemSpace::native>;

//---------------------------------------------------------------------------//
// KERNEL LAUNCHERS
//---------------------------------------------------------------------------//

// Launch the Livermore photoelectric interaction
void rayleigh_interact(const RayleighDeviceRef&     pointers,
                       const ModelInteractPointers& model);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
