//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalGenData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "CerenkovData.hh"
#include "OpticalDistributionData.hh"
#include "OpticalPropertyData.hh"
#include "ScintillationData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Current sizes of the buffers of distribution data.
 *
 * These sizes are updated by value on the host at each step.
 */
struct OpticalBufferSize
{
    size_type cerenkov{0};
    size_type scintillation{0};
};

//---------------------------------------------------------------------------//
/*!
 * Setup options for optical generation.
 *
 * At least one of cerenkov and scintillation must be enabled.
 */
struct OpticalGenSetup
{
    bool cerenkov{false};  //!< Whether Cerenkov is enabled
    bool scintillation{false};  //!< Whether scintillation is enabled
    size_type capacity{0};  //!< Distribution data buffer capacity

    //! True if valid
    explicit CELER_FUNCTION operator bool() const
    {
        return (cerenkov || scintillation) && capacity > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Immutable problem data for generating optical photon distributions.
 */
template<Ownership W, MemSpace M>
struct OpticalGenParamsData
{
    //// DATA ////

    OpticalGenSetup setup;

    //// METHODS ////

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(setup);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalGenParamsData& operator=(OpticalGenParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        setup = other.setup;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Pre-step data needed to generate optical photon distributions.
 */
struct OpticalPreStepData
{
    units::LightSpeed speed;
    Real3 pos{};
    real_type time{};
    OpticalMaterialId opt_mat;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return speed > zero_quantity() && opt_mat;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Optical photon distribution data.
 *
 * The distributions are stored in separate Cerenkov and scintillation buffers
 * indexed by the current buffer size plus the track slot ID. The data is
 * compacted at the end of each step by removing all invalid distributions. The
 * order of the distributions in the buffers is guaranteed to be reproducible.
 */
template<Ownership W, MemSpace M>
struct OpticalGenStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;
    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    // Pre-step data for generating optical photon distributions
    StateItems<OpticalPreStepData> step;

    // Buffers of distribution data for generating optical primaries
    Items<OpticalDistributionData> cerenkov;
    Items<OpticalDistributionData> scintillation;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return step.size(); }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !step.empty() && !(cerenkov.empty() && scintillation.empty());
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
    OpticalGenSetup const& setup = params.setup;
    if (setup.cerenkov)
    {
        resize(&state->cerenkov, setup.capacity);
    }
    if (setup.scintillation)
    {
        resize(&state->scintillation, setup.capacity);
    }

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
