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
#include "base/Array.hh"
#include "base/ArrayIO.hh"
#include "base/Span.hh"
#include "base/StackAllocatorPointers.hh"
#include "base/Types.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleStatePointers.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "physics/material/MaterialParams.hh"
#include "physics/material/MaterialStatePointers.hh"

// Test helpers
#include "base/HostStackAllocatorStore.hh"
#include "gtest/Test.hh"
#include "random/DiagnosticRngEngine.hh"

namespace celeritas
{
template<class T>
class StackAllocatorView;
class ParticleTrackView;
class MaterialTrackView;
struct Interaction;
struct Secondary;
} // namespace celeritas

namespace celeritas_test
{
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
    //!@{
    //! Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;

    using real_type              = celeritas::real_type;
    using PDGNumber              = celeritas::PDGNumber;
    using MevEnergy              = celeritas::units::MevEnergy;

    using MaterialParams    = celeritas::MaterialParams;
    using MaterialTrackView = celeritas::MaterialTrackView;

    using Interaction            = celeritas::Interaction;
    using ParticleParams         = celeritas::ParticleParams;
    using ParticleTrackView      = celeritas::ParticleTrackView;
    using Real3                  = celeritas::Real3;
    using Secondary              = celeritas::Secondary;
    using SecondaryAllocatorView = celeritas::StackAllocatorView<Secondary>;
    using constSpanSecondaries   = celeritas::Span<const Secondary>;

    using HostSecondaryStore = HostStackAllocatorStore<Secondary>;
    //!@}

  public:
    //!@{
    //! Initialize and destroy
    InteractorHostTestBase();
    ~InteractorHostTestBase();
    //!@}

    //!@{
    //! Set and get material properties
    void                  set_material_params(MaterialParams::Input inp);
    const MaterialParams& material_params() const;
    //!@}

    //!@{
    //! Set and get particle params
    void set_particle_params(const ParticleParams::VecAnnotatedDefs& defs);
    const ParticleParams& particle_params() const;
    //!@}

    //!@{
    //! Material properties
    void               set_material(const std::string& name);
    MaterialTrackView& material_track()
    {
        REQUIRE(mt_view_);
        return *mt_view_;
    }
    //!@}

    //!@{
    //! Incident particle properties and access
    void                     set_inc_particle(PDGNumber n, MevEnergy energy);
    void                     set_inc_direction(const Real3& dir);
    const Real3&             direction() const { return inc_direction_; }
    const ParticleTrackView& particle_track() const
    {
        REQUIRE(pt_view_);
        return *pt_view_;
    }
    //!@}

    //!@{
    //! Secondary stack storage and access
    void                             resize_secondaries(int count);
    const HostSecondaryStore& secondaries() const { return secondaries_; }
    SecondaryAllocatorView& secondary_allocator()
    {
        REQUIRE(sa_view_);
        return *sa_view_;
    }
    //!@}

    //!@{
    //! Random number generator
    RandomEngine& rng() { return rng_; }
    //!@}

    // Check for momentum and energy conservation
    void check_conservation(const Interaction& interaction) const;

  private:
    std::shared_ptr<MaterialParams> material_params_;
    std::shared_ptr<ParticleParams> particle_params_;
    RandomEngine                    rng_;

    celeritas::MaterialTrackState     mat_state_;
    std::vector<real_type>            mat_element_scratch_;
    celeritas::MaterialParamsPointers mp_pointers_;
    celeritas::MaterialStatePointers  ms_pointers_;

    celeritas::ParticleTrackState     particle_state_;
    celeritas::ParticleParamsPointers pp_pointers_;
    celeritas::ParticleStatePointers  ps_pointers_;
    Real3                             inc_direction_ = {0, 0, 1};
    HostSecondaryStore                secondaries_;

    // Views
    std::shared_ptr<MaterialTrackView>      mt_view_;
    std::shared_ptr<ParticleTrackView>      pt_view_;
    std::shared_ptr<SecondaryAllocatorView> sa_view_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test
