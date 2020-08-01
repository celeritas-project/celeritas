//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractorHostTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>
#include <vector>
#include "gtest/Test.hh"
#include "base/Array.hh"
#include "base/ArrayIO.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleStatePointers.hh"
#include "physics/base/SecondaryAllocatorPointers.hh"
#include "HostDebugSecondaryStorage.hh"

namespace celeritas
{
class SecondaryAllocatorView;
class ParticleTrackView;
struct Interaction;
struct Secondary;
} // namespace celeritas

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for a host-side Interactor.
 *
 * This class initializes host versions of some of the common inputs to an
 * Interactor. It \b cannot be used for testing device instantiations.
 */
class InteractorHostTestBase : public celeritas::Test
{
  public:
    //@{
    //! Type aliases
    using Real3                = celeritas::Real3;
    using Interaction          = celeritas::Interaction;
    using RandomEngine         = std::mt19937;
    using constSpanSecondaries = span<const Secondary>;
    //@}

  public:
    //@{
    //! Initialize and destroy
    InteractorHostTestBase();
    ~InteractorHostTestBase();
    //@}

    //@{
    //! Set and get particle params
    void set_particle_params(const ParticleParams::VecAnnotatedDefs& defs);
    const ParticleParams& particle_params() const;
    //@}

    //@{
    //! Incident particle properties and access
    void                     set_inc_particle(PDGNumber n, real_type energy);
    void                     set_inc_direction(const Real3& dir);
    const Real3&             direction() const { return inc_direction_; }
    const ParticleTrackView& particle_track() const
    {
        REQUIRE(pt_view_);
        return *pt_view_;
    }
    //@}

    //@{
    //! Secondary stack storage and access
    void                             resize_secondaries(int count);
    const HostDebugSecondaryStorage& secondaries() const
    {
        return secondaries_;
    }
    SecondaryAllocatorView& secondary_allocator()
    {
        REQUIRE(sa_view_);
        return *sa_view_;
    }
    //@}

    //@{
    //! Random number generator
    RandomEngine& rng() { return rng_; }
    //@}

    // Check for momentum and energy conservation
    void check_conservation(const Interaction& interaction) const;

  private:
    std::shared_ptr<ParticleParams> particle_params_;
    RandomEngine                    rng_;

    ParticleTrackState        particle_state_;
    ParticleParamsPointers    pp_pointers_;
    ParticleStatePointers     ps_pointers_;
    Real3                     inc_direction_ = {0, 0, 1};
    HostDebugSecondaryStorage secondaries_;

    // Views
    std::shared_ptr<ParticleTrackView>      pt_view_;
    std::shared_ptr<SecondaryAllocatorView> sa_view_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
