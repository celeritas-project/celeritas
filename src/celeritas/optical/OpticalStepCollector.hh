//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalStepCollector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/user/StepInterface.hh"

#include "OpticalDistributionData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Callback class derived from `StepInterface` that returns StepPoint data for
 * Cerenkov and Scintillation distribution classes.
 */
class OpticalStepCollector final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPParticleParams = std::shared_ptr<ParticleParams const>;
    using MevEnergy = units::MevEnergy;
    using UPStepCollectorDataArray
        = std::unique_ptr<OpticalStepCollectorData[]>;
    using LightSpeed = units::LightSpeed;
    //!@}

    // Construct with ParticleParams and data
    OpticalStepCollector(SPParticleParams particle_params,
                         unsigned int num_streams);

    // Collect optical step data on the host
    void process_steps(HostStepState state) final;

    // Collect optical step data on device
    void process_steps(DeviceStepState state) final;

    // Filtering is *NOT* implemented
    Filters filters() const final { return {}; }

    // Data selection is *NOT* implemented
    StepSelection selection() const final { return {}; }

    // Fetch collected steps
    inline CELER_FUNCTION OpticalStepCollectorData const& get(int tid) const;

  private:
    SPParticleParams particles_;
    UPStepCollectorDataArray step_data_;

    //// HELPER FUNCTIONS ////

    // Calculate speed [1/c]
    inline CELER_FUNCTION LightSpeed speed(ParticleId pid,
                                           MevEnergy energy) const;

    // Step gather loop for both host and device cases
    template<class T>
    void process(T state);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
 * There is currently no access to particle state data to construct and use
 * \c ParticleTrackView::speed() . This is a duplicate code from it.
 */
CELER_FUNCTION units::LightSpeed
OpticalStepCollector::speed(ParticleId pid, MevEnergy energy) const
{
    auto const& p = particles_->get(pid);
    real_type const mcsq = p.mass().value();
    real_type inv_gamma = mcsq / (energy.value() + mcsq);
    real_type beta_sq = 1 - ipow<2>(inv_gamma);

    return LightSpeed{std::sqrt(beta_sq)};
}

//---------------------------------------------------------------------------//
/*!
 * Get optical step data given a stream id.
 */
CELER_FUNCTION OpticalStepCollectorData const&
OpticalStepCollector::get(int tid) const
{
    auto const& opt_step = step_data_[tid];
    CELER_ENSURE(opt_step);
    return opt_step;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
