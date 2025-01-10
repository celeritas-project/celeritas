//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OffloadData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

#include "CherenkovData.hh"
#include "GeneratorDistributionData.hh"
#include "ScintillationData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Current sizes of the buffers of distribution data.
 *
 * These sizes are updated by value on the host at each core step.
 */
struct OffloadBufferSize
{
    size_type cherenkov{0};
    size_type scintillation{0};
    size_type num_photons{0};
};

//---------------------------------------------------------------------------//
/*!
 * Setup options for optical generation.
 *
 * At least one of cherenkov and scintillation must be enabled.
 */
struct OffloadOptions
{
    bool cherenkov{false};  //!< Whether Cherenkov is enabled
    bool scintillation{false};  //!< Whether scintillation is enabled
    size_type capacity{0};  //!< Distribution data buffer capacity

    //! True if valid
    explicit CELER_FUNCTION operator bool() const
    {
        return (cherenkov || scintillation) && capacity > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Immutable problem data for generating optical photon distributions.
 */
template<Ownership W, MemSpace M>
struct OffloadParamsData
{
    //// DATA ////

    OffloadOptions setup;

    //// METHODS ////

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(setup);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OffloadParamsData& operator=(OffloadParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        setup = other.setup;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Pre-step data needed to generate optical photon distributions.
 *
 * If the optical material is not set, the other properties are invalid.
 */
struct OffloadPreStepData
{
    units::LightSpeed speed;
    Real3 pos{};
    real_type time{};
    OpticalMaterialId material;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return material && speed > zero_quantity();
    }
};

//---------------------------------------------------------------------------//
/*!
 * Optical photon distribution data.
 *
 * The distributions are stored in separate Cherenkov and scintillation buffers
 * indexed by the current buffer size plus the track slot ID. The data is
 * compacted at the end of each step by removing all invalid distributions. The
 * order of the distributions in the buffers is guaranteed to be reproducible.
 */
template<Ownership W, MemSpace M>
struct OffloadStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;
    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    // Pre-step data for generating optical photon distributions
    StateItems<OffloadPreStepData> step;

    // Buffers of distribution data for generating optical photons
    Items<optical::GeneratorDistributionData> cherenkov;
    Items<optical::GeneratorDistributionData> scintillation;

    // Determines which distribution a thread will generate a primary from
    Items<size_type> offsets;

    //// METHODS ////

    //! Number of states
    CELER_FUNCTION size_type size() const { return step.size(); }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !step.empty() && !offsets.empty()
               && !(cherenkov.empty() && scintillation.empty());
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OffloadStateData& operator=(OffloadStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        step = other.step;
        cherenkov = other.cherenkov;
        scintillation = other.scintillation;
        offsets = other.offsets;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize optical states.
 */
template<MemSpace M>
void resize(OffloadStateData<Ownership::value, M>* state,
            HostCRef<OffloadParamsData> const& params,
            StreamId,
            size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);

    resize(&state->step, size);
    OffloadOptions const& setup = params.setup;
    if (setup.cherenkov)
    {
        resize(&state->cherenkov, setup.capacity);
    }
    if (setup.scintillation)
    {
        resize(&state->scintillation, setup.capacity);
    }
    resize(&state->offsets, setup.capacity);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
