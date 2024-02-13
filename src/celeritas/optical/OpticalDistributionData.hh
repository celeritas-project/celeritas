//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalDistributionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/EnumArray.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Pre- and post-step data for sampling optical photons.
 */
struct OpticalStepData
{
    units::LightSpeed speed{};
    Real3 pos{};

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const { return speed.value() > 0; }
};

//---------------------------------------------------------------------------//
/*!
 * Input data for sampling optical photons.
 */
struct OpticalDistributionData
{
    size_type num_photons{};  //!< Sampled number of photons to generate
    real_type time{};  //!< Pre-step time
    real_type step_length{};
    units::ElementaryCharge charge;
    OpticalMaterialId material;
    EnumArray<StepPoint, OpticalStepData> points;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_photons > 0 && step_length > 0 && material;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data used by \c OpticalStepCollector class implementation.
 */
struct OpticalStepCollectorData
{
    real_type time{};  //!< Pre-step time
    real_type step_length{};  //!< Step length
    EnumArray<StepPoint, OpticalStepData> points;  //!< Pre- and post-steps

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return step_length > 0 && points[StepPoint::pre].speed.value() > 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Shared optical data for optical generator classes such as
 * \c CerenkovGenerator and \c ScintillationGenerator .
 */
template<Ownership W, MemSpace M>
struct OpticalInitStateData
{
    template<class T>
    using Items = Collection<T, W, M>;

    //// MEMBER DATA ////

    Items<OpticalDistributionData> distributions;

    //// MEMBER FUNCTIONS ////

    //! Whether all data are assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !distributions.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    OpticalInitStateData& operator=(OpticalInitStateData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        distributions = other.distributions;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
