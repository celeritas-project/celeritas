//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/WavelengthShiftData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/StateDataStore.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/phys/GeneratorInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Input data for sampling WLS optical photons.
 */
struct WlsDistributionData
{
    GeneratorType type{GeneratorType::size_};
    size_type num_photons{};  //!< Sampled number of photons to generate
    units::MevEnergy energy;
    real_type time{};  //!< Post-step time
    Real3 position{};
    PrimaryId primary;  //!< For correlating to G4 tracks
    OptMatId material;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return (type == GeneratorType::wls || type == GeneratorType::wls2)
               && num_photons > 0 && energy > zero_quantity() && material;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Material dependent scalar property of wavelength shift (WLS).
 */
struct WlsMaterialRecord
{
    real_type mean_num_photons{};  //!< Mean number of reemitted photons
    real_type time_constant{};  //!< Time delay of WLS [time]

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return mean_num_photons > 0 && time_constant > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Wavelength shift data
 */
template<Ownership W, MemSpace M>
struct WavelengthShiftData
{
    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OpticalMaterialItems = Collection<T, W, M, OptMatId>;

    //// MEMBER DATA ////

    GeneratorType type{GeneratorType::size_};
    OpticalMaterialItems<WlsMaterialRecord> wls_record;

    // Cumulative probability of emission as a function of energy
    OpticalMaterialItems<NonuniformGridRecord> energy_cdf;

    // Time profile model
    WlsDistribution time_profile{WlsDistribution::size_};

    // Backend data
    Items<real_type> reals;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return (type == GeneratorType::wls || type == GeneratorType::wls2)
               && !wls_record.empty() && !energy_cdf.empty()
               && time_profile != WlsDistribution::size_;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    WavelengthShiftData& operator=(WavelengthShiftData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        type = other.type;
        wls_record = other.wls_record;
        energy_cdf = other.energy_cdf;
        time_profile = other.time_profile;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Optical wavelength shift distribution data.
 *
 * The distributions are stored in a buffer indexed by the current buffer size
 * plus the track slot ID. The data is compacted at the end of each step by
 * removing all invalid distributions. The order of the distributions in the
 * buffers is guaranteed to be reproducible.
 */
template<Ownership W, MemSpace M>
struct WlsGeneratorStateData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;

    //// DATA ////

    // Buffer of distribution data for generating optical photons
    Items<WlsDistributionData> distributions;

    // Determines which distribution a thread will generate a primary from
    Items<size_type> offsets;

    //// METHODS ////

    //! State size
    CELER_FUNCTION size_type size() const { return distributions.size(); }

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !distributions.empty() && distributions.size() == offsets.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    WlsGeneratorStateData& operator=(WlsGeneratorStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        distributions = other.distributions;
        offsets = other.offsets;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store optical WLS generation states in aux data.
 */
template<MemSpace M>
struct WlsGeneratorState : public GeneratorStateBase
{
    StateDataStore<WlsGeneratorStateData, M> store;

    //! True if states have been allocated
    explicit operator bool() const { return static_cast<bool>(store); }
};

//---------------------------------------------------------------------------//
/*!
 * Resize optical buffers.
 */
template<MemSpace M>
void resize(
    WlsGeneratorStateData<Ownership::value, M>* state, StreamId, size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&state->distributions, size);
    resize(&state->offsets, size);
    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
