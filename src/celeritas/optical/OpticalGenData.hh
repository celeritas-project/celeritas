//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalStepData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/StackAllocatorData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/CerenkovData.hh"
#include "celeritas/optical/OpticalDistributionData.hh"
#include "celeritas/optical/OpticalPropertyData.hh"
#include "celeritas/optical/ScintillationData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Immutable problem data for generating optical photon distributions.
 */
template<Ownership W, MemSpace M>
struct OpticalGenParamsData
{
    //// DATA ////

    // Distribution data stack storage per state size
    real_type distribution_stack_factor = 1;

    NativeCRef<ScintillationData> scintillation;
    NativeCRef<CerenkovData> cerenkov;
    NativeCRef<OpticalPropertyData> properties;

    //// METHODS ////

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return distribution_stack_factor > 0
               && (scintillation || (cerenkov && properties));
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalGenParamsData& operator=(OpticalGenParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        distribution_stack_factor = other.distribution_stack_factor;
        scintillation = other.scintillation;
        cerenkov = other.cerenkov;
        properties = other.properties;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Pre-step data needed for generating optical photon distributions.
 */
struct OpticalPreStepData
{
    units::LightSpeed speed;  //!< Pre-step speed
    Real3 pos{};  //!< Pre-step position
    real_type time{};  //!< Pre-step time

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return speed > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Optical photon distribution data and primaries.
 */
template<Ownership W, MemSpace M>
struct OpticalGenStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;
    using DistributionStackData
        = StackAllocatorData<OpticalDistributionData, W, M>;

    //// DATA ////

    // Pre-step data for generating optical photon distributions
    StateItems<OpticalPreStepData> step;

    // Buffers of distribution data for generating optical primaries
    DistributionStackData cerenkov;
    DistributionStackData scintillation;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return step.size(); }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !step.empty() && (cerenkov || scintillation);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalGenStateData& operator=(OpticalGenStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        step = other.step;
        cerenkov = other.cerenkov;
        scintillation = other.scintillation;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize optical states.
 */
template<MemSpace M>
void resize(OpticalGenStateData<Ownership::value, M>* state,
            HostCRef<OpticalGenParamsData> const& params,
            StreamId,
            size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);

    resize(&state->step, size);
    size_type stack_size = size * params.distribution_stack_factor;
    if (params.cerenkov)
    {
        resize(&state->cerenkov, stack_size);
    }
    if (params.scintillation)
    {
        resize(&state->scintillation, stack_size);
    }

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
